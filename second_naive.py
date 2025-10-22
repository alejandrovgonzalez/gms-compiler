import os
import signal
import time
from natsort import natsorted
from fractions import Fraction
from benchmark_utils import GMS, calculate_circuit_runtime

def second_naive_sqgs_for_gms(thetas):
    """
    Helper function for the second naive algorithm.
    Returns where to place single-qubit gates when making a gms.
    """
    hadamards = []
    rxs = []
    for i in range(len(thetas)):
        for j in range(len(thetas[0])):
            if thetas[i][j] != 0:
                hadamards.append(i)
                rxs.append(i)
                rxs.append(j)
    return hadamards, rxs

def second_naive_get_qubits(in_path: str):
    """
    Helper function for the second naive algorithm.
    Gives us total qubit count and an address book for circuits with multiple qregs.
    """
    qreg: Dict[str, dict] = {} 
    total_qubit_count = 0
    with open(in_path) as circuit:
        for inst in circuit:
            if inst.startswith('qreg'):
                name = inst[inst.find(' ')+1:inst.find('[')]
                qubit_count = int(inst[inst.find('[')+1: inst.find(']')])
                qreg[name] = {}
                for i in range(qubit_count):
                    qreg[name].setdefault(i, total_qubit_count)
                    total_qubit_count += 1

    return qreg, total_qubit_count

def second_naive_save_gms(saved_circuit, thetas):
    """
    Helper function for the second naive algorithm.
    Saves a GMS into the circuit given thetas, plus also writes the needed SQGs.
    """
    gms = GMS(thetas)
    hadamards, rxs = second_naive_sqgs_for_gms(thetas)
    qubits = len(thetas)
    for i in hadamards:
        saved_circuit.write('h q[' + str(i) + '];\n')
    saved_circuit.write(gms.to_qasm()+'\n')
    for i in rxs:
        saved_circuit.write('rx(0.5*pi) q[' + str(i) + '];\n')
    for i in hadamards:
        saved_circuit.write('h q[' + str(i) + '];\n')


def second_naive_algorithm(out_path: str, in_path: str):

    qreg, total_qubit_count = second_naive_get_qubits(in_path)

    with open(out_path,'w') as saved_circuit:
        with open(in_path) as circuit:
            thetas = [[0 for _ in range(total_qubit_count)] for _ in range(total_qubit_count)]
            for inst in circuit:
                if inst.startswith('OPENQASM') or inst.startswith('include') or\
                    inst.startswith('creg') or inst.startswith('measure') or\
                    inst.startswith('barrier') or inst.startswith('//') or\
                    inst.startswith('\n'):
                    saved_circuit.write(inst)
                    if inst.startswith('include'):
                        saved_circuit.write('qreg q['+ str(total_qubit_count) + '];\n')
                elif inst.startswith('qreg'):
                    continue
                elif inst.startswith('cx'):
                    pos1 = inst.find('[')
                    pos2 = inst[pos1+1:].find('[')+1+pos1
                    q1 = int(inst[pos1+1: inst.find(']')])
                    q2 = int(inst[pos2+1: inst[pos2:].find(']')+pos2])
                    name1 = inst[inst.find(' ')+1:pos1]
                    name2 = inst[inst.find(',')+1:pos2]
                    q1 = qreg[name1][q1]
                    q2 = qreg[name2][q2]
                    
                    if any(thetas[q1]) or any(thetas[q2]) or any([row[q1] for row in thetas]) or\
                         any([row[q2] for row in thetas]):
                         # incoming CNOT has overlap in the qubits with current GMS
                         second_naive_save_gms(saved_circuit, thetas)
                         thetas = [[0 for _ in range(total_qubit_count)] for _ in range(total_qubit_count)]

                    thetas[q1][q2] = Fraction(-1,2)
                elif inst.startswith('u3') or inst.startswith('u2') or inst.startswith('rx') or\
                inst.startswith('rz') or inst.startswith('x ') or inst.startswith('sx') or\
                inst.startswith('u1') or inst.startswith('h'):
                    if any([any(thetas[i]) for i in range(len(thetas))]):
                        second_naive_save_gms(saved_circuit, thetas)
                        thetas = [[0 for _ in range(total_qubit_count)] for _ in range(total_qubit_count)]
                    
                    name = inst[inst.find(' ')+1:inst.find('[')] 
                    q = int(inst[inst.find('[')+1: inst.find(']')])
                    new_inst = inst[:inst.find(' ')] + ' q[' + str(qreg[name][q]) + '];\n'
                    saved_circuit.write(new_inst)
            if any([any(thetas[i]) for i in range(len(thetas))]):
                second_naive_save_gms(saved_circuit, thetas)

def compile_second_naive(in_path: str, out_path: str):
    """
    Run the second naive algorithm we developed on the input circuits.
    """
    filenames = natsorted([filename for filename in os.listdir(in_path)])
    if filenames[0].find('opt0') != -1:
        # It's an MQTBench circuit
        for filename in filenames[:]:
            idx = filename.find('opt0')
            start = filename[idx:].find('_')
            end = filename[idx+start+1:].find('.')
            qubits = int(filename[idx+start+1:idx+start+end+1])
            # Remove larger circuits
            if qubits > 25:
                filenames.remove(filename)
            elif filename.startswith('qnn_nativegates_ibm_qiskit_opt0_'):
                filenames.remove(filename)
    with open(os.path.join(out_path, 'exec_times.csv'),"a") as fresult:
        fresult.write("Circuit, Time \n")
        for filename in filenames:
            # Remove other large (deep) circuits as we are just testing the naive algorithm out
            if filename.startswith('bwt_n37') or filename.startswith('bwt_n57') or\
            filename.startswith('bwt_n97') or filename.startswith('vqe_n24') or\
            filename.startswith('square_root_n60') or filename.startswith('C2H4_UCCSD_JW_sto3g') or\
            filename.startswith('multiplier_n350') or filename.startswith('multiplier_n400'):
                continue
            signal.alarm(900) # 15 min timeout
            start = time.time()
            try:
                second_naive_algorithm(os.path.join(out_path, filename), os.path.join(in_path, filename))
            except TimeoutError:
                print("TO'd for", filename)
                fresult.write( ','.join([filename, "Timeout"] ) + '\n')
                continue
            end = time.time()
            signal.alarm(0)

            fresult.write(';'.join([filename, str(end - start)]) + '\n')

            print("Finished for", filename)

def save_time_second_naive(original_path, second_naive_path, out_path):
    """
    Calculate and save the quantum time and the gate counts of the circuits resulting from the first and second naive algorithms.
    """
    with open(out_path, 'w') as out_file:
        out_file.write(','.join(['', '', 'Original circuit', '', '', 'Second Naive Algorithm', '','\n']))
        out_file.write(','.join(['Circuit', 'SQG', 'Entangling', 'T', 'SQG', 'Entangling', 'T\n']))
        for file in natsorted(os.listdir(original_path)):
            if not file.endswith('.qasm') or not os.path.isfile(os.path.join(second_naive_path, file)):
                print('Skipping', file)
                continue

            time_original, sqg_original, tqg_original = calculate_circuit_runtime(os.path.join(original_path,file))
            time_second_naive, sqg_second_naive, tqg_second_naive = calculate_circuit_runtime(os.path.join(second_naive_path,file))

            out_file.write(','.join([file, str(sqg_original), str(tqg_original), str(time_original),
                                    str(sqg_second_naive), str(tqg_second_naive), str(time_second_naive),])+'\n')

            print('Done', file)