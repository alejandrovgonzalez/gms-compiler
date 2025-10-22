import os, pickle, signal, time, csv, re, sys
import matplotlib.pyplot as plt
from typing import Dict, List
import pyzx as zx
from pyzx import Circuit
from pyzx.circuit.gates import CNOT
from pyzx.circuit import Gate, HAD
from pyzx.utils import FractionLike
import numpy as np
from fractions import Fraction
import random
from natsort import natsorted

GATE_TIMES = {'R': 110, 'rxx': 672, 'gms': 672} # In microseconds

class HCNOTH(CNOT):
    def __init__(self, control: int, target: int, h_left: bool, h_right: bool):
        super().__init__(control, target)
        self.h_left = h_left
        self.h_right = h_right

class GMS(Gate):
    name = 'GMS'
    qasm_name = 'gms'
    
    def __init__(self, thetas: List[List[FractionLike]]) -> None:
        self.thetas = thetas

    def to_qasm(self) -> str:
        """
        Return custom .qasm representation of GMS
        """
        phases = []
        for i in range(len(self.thetas)-1):
            phases.append(self.thetas[i][i+1:])
        return self.qasm_name + '({});'.format(str(phases))

    @classmethod
    def from_qasm(cls, inst):
        """
        Parse custom .qasm instruction into a GMS class instance.
        """
        inst = inst[5:-3] # leave just the list of lists in the input .qasm instruction
        thetas = []
        idx2 = 0
        while (idx1 := inst[idx2:].find('[')) != -1:
            idx1 = idx1+idx2
            idx2 = inst[idx1:].find(']')+idx1
            row = []
            #Regex matches commas that are outside parenthesis, we split on those matches
            for entry in re.split(r',\s*(?![^()]*\))', inst[idx1+1: idx2]):
                if (par := entry.find('(')) != -1:
                    #non-zero entry in the form of Fraction(num,den)
                    num, den = entry[par+1: -1].split(', ')
                    row.append(Fraction(int(num), int(den)))
                else:
                    row.append(int(entry))
            thetas.append(row)
        qubit_count = max(len(x) for x in thetas) + 1
        thetas = [ [0]*(qubit_count-len(row)) + row for row in thetas ]
        return GMS(thetas)

    def involved_qubits(self) -> set:
        """
        Return set of qubits involved in global gate
        """
        qubits = set()
        for i in range(len(self.thetas)):
            for j in range(i+1,len(self.thetas[i])):
                if (theta := self.thetas[i][j]) != 0:
                    qubits.add(i)
                    qubits.add(j)
        return qubits

class Global_gate_handler():
    def __init__(self):
        self.global_gates: List = []
        self.cz_global_gates: List = []
        self.current_global: List = []
        self.status_qubits: list[tuple[set[int], Gate]] = []
        self.global_controls: set[int] = set()
        self.global_targets: set[int] = set()
        self.busy_qubits: set[int] = set()
        self.q_tracker: Dict[int, list[int]] = {} #To track cnots working on each qubit for every layer of cnots

    def add_cnot_to_current_global(self, cnot) -> None:
        """
        Adds input CNOT gate to the tracking list for the current global gate.
        """
        self.current_global.append(cnot)
        self.global_controls.add(cnot.control)
        self.global_targets.add(cnot.target)
    
    def fits_in_current_global(self, cnot) -> bool:
        """
        Checks if `cnot` fits in the current global gate.
        """
        next_in_control = next((gate for gate in reversed(self.current_global) if gate.control == cnot.control or gate.target == cnot.control), None)
        next_in_target = next((gate for gate in reversed(self.current_global) if gate.control == cnot.target or gate.target == cnot.target), None)

        control_ok = cnot.control not in self.busy_qubits and (next_in_control is None or (not next_in_control.target == cnot.control and cnot.h_right == next_in_control.h_left))
        target_ok = cnot.target not in self.busy_qubits and (next_in_target is None or next_in_target.target == cnot.target or not next_in_target.h_left)
        
        return control_ok and target_ok

    def classify_cnot(self, idx: int, cnots: list[CNOT], good_qubits: list[int]) -> tuple[bool,bool]:
        """
        Decides how the `cnot` would look like as an RXX + Rx + possibly a Hadamard.
        It might take a Hadamard away from `current_global`
        """
        cnot  = cnots[idx]
        i = self.q_tracker[cnot.control].index(idx)

        is_last = i==len(self.q_tracker[cnot.control])-1 # More CNOTs to the left?
        h_already_in_left = is_last and cnot.control not in good_qubits # If there is Hadamard from advancing the frontier it will be pushed (no Hadamard to the left)
        prev_is_target = not is_last and cnots[self.q_tracker[cnot.control][i+1]].target == cnot.control
        h_left = prev_is_target or h_already_in_left

        if i == 0: # no CNOTs to the right in `cnots`, need to check `current_global`
            next_in_line = next((gate for gate in reversed(self.current_global) if gate.control == cnot.control or gate.target == cnot.control), None)
            if next_in_line is not None and next_in_line.control == cnot.control and next_in_line.h_left and (cnot.control not in self.busy_qubits):
                # Two Hadamards cancel out
                h_right = False
                next_in_line.h_left = False
            else:
                h_right = True
        else:
            h_right = cnots[self.q_tracker[cnot.control][i-1]].target == cnot.control

        return h_left, h_right

    def update_global_gates(self):
        self.global_gates.append(self.current_global)

    def is_used_in_global(self, q):
        """
        Checks if a qubit is being used in the current_global gate
        """
        for cnot in self.current_global:
            if cnot.control == q or cnot.target == q:
                return True
        return False

    def can_be_placed(self, q, g):
        """
        Checks if gate `g` in qubit `q` can be placed by looking at the other delayed gates
        """
        global_ok = not self.is_used_in_global(q)
        pending_ok = True
        if global_ok:
            for gate in reversed(self.status_qubits):
                matched = q in gate[0]
                name = gate[1].name
                if g == 'HAD':
                    commutes = name == 'HAD'
                else:
                    # ZPhase and CZ
                    commutes = name == 'ZPhase' or name == 'CZ'
                if matched and not commutes:
                    pending_ok = False
                    break

        return global_ok and pending_ok

    def flush_variables(self):
        self.status_qubits, self.current_global = [], []
        self.global_controls, self.global_targets, self.busy_qubits = set(), set(), set()
    
    def finish_layer(self, hads):
        """
        Delays placing the Hadamards resulting from clearing a frontier layer.
        """
        self.q_tracker = {}
        for q in hads:
            self.status_qubits.append( ({q},HAD(target=q)) )
            self.busy_qubits.add(q)

def load_circuit(in_path, filename):
    """
    Load input circuit into a PyZX graph
    """
    try:
        input_circuit = zx.Circuit.load(os.path.join(in_path, filename)).to_basic_gates()
    except:
        print("Failed PyZX QASM load for", filename)
        return None, None
    original_2qubit_count = input_circuit.twoqubitcount() # For statistics
    g = input_circuit.to_graph()
    signal.alarm(900) # 15 min timeout
    start = time.time()
    try:
        zx.full_reduce(g, quiet=True)
        g.normalize()
    except TimeoutError:
        print("PyZX failed parsing", filename)
        input_circuit, g = None, None
    end = time.time()
    signal.alarm(0)
    return input_circuit, g
    

def save_recompiled(out_path: str, filename: str, resultCircuit: Circuit):
    """
    Saves a "recompiled" version of the circuit, in which we write a custom 'gms'
    instruction onto the file instead of RXX gates.
    """
    with open(os.path.join(out_path,'recompiled', filename),'w') as saved_circuit:
        thetas = [[0 for _ in range(resultCircuit.qubits)] for _ in range(resultCircuit.qubits)]
        for inst in resultCircuit.to_qasm().splitlines():
            if inst.startswith('rxx'):
                pos1 = inst.find('[')
                pos2 = inst[pos1+1:].find('[')+1+pos1
                q1 = int(inst[pos1+1: inst.find(']')])
                q2 = int(inst[pos2+1: inst[pos2:].find(']')+pos2])

                if thetas[q1][q2] != 0 or thetas[q2][q1] != 0:
                    # This would lead to different angles in some XX gates in the GMS, which we don't allow.
                    gms = GMS(thetas)
                    saved_circuit.write(gms.to_qasm()+'\n')
                    thetas = [[0 for _ in range(resultCircuit.qubits)] for _ in range(resultCircuit.qubits)]

                thetas[q1][q2] = Fraction(-1,2)
                thetas[q2][q1] = Fraction(-1,2)
            else:
                if any([any(thetas[i]) for i in range(len(thetas))]):
                    gms = GMS(thetas)
                    saved_circuit.write(gms.to_qasm()+'\n')
                    thetas = [[0 for _ in range(resultCircuit.qubits)] for _ in range(resultCircuit.qubits)]
                saved_circuit.write(inst+'\n')
        if any([any(thetas[i]) for i in range(len(thetas))]):
            gms = GMS(thetas)
            saved_circuit.write(gms.to_qasm()+'\n')

def calculate_circuit_runtime(path: str):
    """
    Calculate circuit runtime for input .qasm `circuit` . Also returns SQG and entangling gate counts.
    """
    # Accumulated time for each qubit
    qreg: Dict[str, dict] = {} 
    sqg = 0
    tqg = 0
    with open(path) as circuit:
        for inst in circuit:
            if inst.startswith('OPENQASM') or inst.startswith('include') or\
                inst.startswith('creg') or inst.startswith('measure') or\
                inst.startswith('barrier') or inst.startswith('//') or\
                inst.startswith('\n'):
                continue
            if inst.startswith('qreg'):
                name = inst[inst.find(' ')+1:inst.find('[')]
                qubit_count = int(inst[inst.find('[')+1: inst.find(']')])
                qreg[name] = dict.fromkeys(range(qubit_count),0)
            elif inst.startswith('rxx'):
                tqg += 1

                max_t = 0
                for _, v in qreg.items():
                    for _, t in v.items():
                        max_t = max(max_t, t)
                t_after_rxx = max_t + GATE_TIMES['rxx']
                for reg, v in qreg.items():
                    for q, t in v.items():
                        qreg[reg][q] = t_after_rxx

            elif inst.startswith('u3') or inst.startswith('u2'):
                # For quantum chemistry circuits.
                # they can potentially cost two hardware rotations but here
                # we conservatively give them the cost of 1.
                # This just makes the comparison with the input circuit "look worse" for us.
                sqg += 1
                name = inst[inst.find(' ')+1:inst.find('[')] 
                q = int(inst[inst.find('[')+1: inst.find(']')])
                qreg[name][q] += GATE_TIMES['R']

            elif inst.startswith('rx') or inst.startswith('rz')\
                or inst.startswith('x ') or inst.startswith('sx')\
                or inst.startswith('u1'):
                sqg += 1
                name = inst[inst.find(' ')+1:inst.find('[')] 
                q = int(inst[inst.find('[')+1: inst.find(']')])
                qreg[name][q] += GATE_TIMES['R']
            elif inst.startswith('h'):
                sqg += 2
                name = inst[inst.find(' ')+1:inst.find('[')]
                q = int(inst[inst.find('[')+1: inst.find(']')])
                qreg[name][q] += GATE_TIMES['R'] * 2
            elif inst.startswith('gms'):
                tqg += 1

                # In circuits with gms we only have one qreg, labelled 'q'
                max_t = 0
                for _, t in qreg['q'].items():
                    max_t = max(max_t, t)
                t_after_gms = max_t + GATE_TIMES['gms']

                for q, t in qreg['q'].items():
                    qreg['q'][q] = t_after_gms

            elif inst.startswith('cx'):
                tqg += 1
                sqg += 6 # Two hadamards and two Rx

                pos1 = inst.find('[')
                pos2 = inst[pos1+1:].find('[')+1+pos1
                q1 = int(inst[pos1+1: inst.find(']')])
                q2 = int(inst[pos2+1: inst[pos2:].find(']')+pos2])
                name1 = inst[inst.find(' ')+1:pos1]
                name2 = inst[inst.find(',')+1:pos2]

                t1 = qreg[name1][q1]
                t2 = qreg[name2][q2]

                max_t = 0
                for _, v in qreg.items():
                    for q, t in v.items():
                        max_t = max(max_t, t)
                max_t = max(max_t, t1 + GATE_TIMES['R']*2) # control conjugated by hadamards
                t_after_rxx = max_t + GATE_TIMES['rxx']
                for reg, v in qreg.items():
                    for q, t in v.items():
                        qreg[reg][q] = t_after_rxx
                qreg[name1][q1] += GATE_TIMES['R']*3 #Control conjugated by hadamards plus Rx
                qreg[name2][q2] += GATE_TIMES['R']  # Rx in target
    final_time = float("{:.1f}".format(max(max(reg.values()) for reg in qreg.values())/1000 )) #in milliseconds
    return final_time, sqg, tqg

def save_time_comparisons(original_path, ilp_path, peephole_path, qiskit_path, out_path):
    """
    Calculate and save the quantum time and the gate counts of input circuits, and save their comparison in out_path
    """
    with open(out_path, 'w') as out_file:
        out_file.write(','.join(['', '', 'Original circuit', '', '', 'Peephole', '', '', 'Peephole + LP', '', '', 'Qiskit', '\n']))
        out_file.write(','.join(['Circuit', 'SQG', 'Entangling', 'T', 'SQG', 'Entangling', 'T', 'SQG', 'Entangling', 'T', 'SQG', 'Entangling', 'T\n']))
        for file in natsorted(os.listdir(original_path)):
            if not file.endswith('.qasm') or not os.path.isfile(os.path.join(ilp_path, file))\
                or not os.path.isfile(os.path.join(peephole_path, file))\
                or not os.path.isfile(os.path.join(qiskit_path, file)):
                print('Skipping', file)
                continue

            time_original, sqg_original, tqg_original = calculate_circuit_runtime(os.path.join(original_path,file))
            time_ilp, sqg_ilp, tqg_ilp = calculate_circuit_runtime(os.path.join(ilp_path,file))
            time_peephole, sqg_peephole, tqg_peephole = calculate_circuit_runtime(os.path.join(peephole_path,file))
            time_qiskit, sqg_qiskit, tqg_qiskit = calculate_circuit_runtime(os.path.join(qiskit_path,file))

            out_file.write(','.join([file, str(sqg_original), str(tqg_original), str(time_original),
                                    str(sqg_peephole), str(tqg_peephole), str(time_peephole),
                                    str(sqg_ilp), str(tqg_ilp), str(time_ilp),
                                    str(sqg_qiskit), str(tqg_qiskit), str(time_qiskit),])+'\n')

            print('Done', file)

def timeout_handler(_, __):
    raise TimeoutError

def remove_meas_and_barriers(path):
    """
    I used this to remove measurements and barriers from the circuits that we benchmark
    """
    for filename in os.listdir(path):
        if not os.path.isfile(os.path.join(path, filename)):
            continue
        with open(path + filename, "r") as file:
            with open(os.path.join(path, 'no_measurements', filename),"w") as output:
                while line := file.readline():
                    if ("measure" not in line) and ("barrier" not in line):
                        output.write(line)

def bold_benchmark_results(in_path, out_path):
    """
    I used this to make the best result in each entry of the benchmark results in bold.
    """
    with open(in_path) as in_csv:
        with open(out_path, mode='w') as out_file:
            csv_reader = csv.reader(in_csv, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0 or line_count == 1:
                    line_count += 1
                    continue
                t_original, t_peephole, t_ilp, t_qiskit = row[3], row[6], row[9], row[12]
                best = str(min(float(t_original), float(t_peephole), float(t_ilp), float(t_qiskit)))
                written_row = [entry if (entry == row[0] or str(float(entry)) != best) else '\\textbf{{ {} }}'.format(entry) for entry in row]
                
                #Let's clean up the circuit names
                if written_row[0].find('opt0') != -1:
                    #MQT circuit
                    entry = written_row[0]
                    name = entry[0:entry.find('_')]
                    idx = entry.find('opt0')
                    start = entry[idx:].find('_')
                    end = entry[idx+start+1:].find('.')
                    qubits = int(entry[idx+start+1:idx+start+end+1])
                    written_row[0] = name + '\\_n' + str(qubits)
                elif written_row[0].find('transpiled') != -1:
                    #QASMbench
                    entry = written_row[0]
                    idx = entry.find('_transpiled.qasm')
                    written_row[0] = written_row[0][:idx]
                    written_row[0] = written_row[0].replace('_','\\_')
                else:
                    #Chemistry
                    entry = written_row[0]
                    idx = entry.find('.qasm')
                    written_row[0] = written_row[0][:idx]
                    written_row[0] = written_row[0].replace('_','\\_')

                out_file.write(' & '.join(written_row)+' \\\\\n')
            out_file.write('\\hline')