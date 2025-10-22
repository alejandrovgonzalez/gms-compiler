import time, signal, os
from typing import Dict, List, Callable
from fractions import Fraction
from pyzx import Circuit
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.extract import (neighbors_of_frontier, remove_gadget, bi_adj,
                          column_optimal_swap, filter_duplicate_cnots,
                          id_simp, graph_to_swaps, connectivity_from_biadj)
from pyzx.circuit.gates import CNOT
from pyzx.utils import EdgeType
from pyzx.linalg import Mat2
from pyzx.circuit import RXX, ZPhase, HAD, XPhase
from extract_LP import create_LP
from benchmark_utils import (HCNOTH, Global_gate_handler, save_recompiled,
                             load_circuit, remove_meas_and_barriers,
                             save_time_comparisons)
from qiskit_synthesis import compile_qiskit
from second_naive import compile_second_naive, save_time_second_naive
from natsort import natsorted

def simplify_sqgs(c: Circuit):
    """
    Performs Hadamard cancellation, fusion of rotation gates,
    and change of rotation axis when beneficial.
    """

    excluded = []
    commute_stack = {k: [] for k in range(c.qubits)}
    for i, g in enumerate(c.gates):
        if i not in excluded:
            if g.name == 'HAD':
                if len(commute_stack[g.target]) == 0:
                    commute_stack[g.target] = [('HAD',i, None, None)]
                elif len(commute_stack[g.target]) == 1:
                    name, idx, phase, rxx = commute_stack[g.target][0]
                    if name == 'HAD':
                        # H-H cancellation
                        excluded.append(commute_stack[g.target][0][1])
                        excluded.append(i)
                        commute_stack[g.target] = []
                    elif name == 'XPhase':
                        c.gates[idx] = XPhase(target=c.gates[idx].target, phase=phase)
                        commute_stack[g.target] = [('HAD',i, None, None)]
                    else:
                        c.gates[idx] = ZPhase(target=c.gates[idx].target, phase=phase)
                        commute_stack[g.target] = [('HAD',i, None, None)]
                else:
                    # Simplify H-RX/Z-H sequences
                    name, idx, phase, rxx = commute_stack[g.target][1]
                    if name == 'XPhase':
                        c.gates[idx] = ZPhase(target=c.gates[idx].target, phase=phase) if not rxx else XPhase(target=c.gates[idx].target, phase=phase)
                    else:
                        c.gates[idx] = XPhase(target=c.gates[idx].target, phase=phase)
                    if not rxx:
                        excluded.append(commute_stack[g.target][0][1])
                        excluded.append(i)
                        commute_stack[g.target] = []
                    else:
                        commute_stack[g.target] = [('HAD',i, None, None)]

            elif g.name == 'XPhase':
                if len(commute_stack[g.target]) > 0:
                    name, idx, phase, rxx = commute_stack[g.target][-1]
                    if name == 'HAD':
                        commute_stack[g.target].append(('XPhase',i, g.phase, False)) #boolean indicates if RXX is in front
                    elif name == 'XPhase':
                        # Merge XPhases
                        commute_stack[g.target][-1] = (name, idx, phase + g.phase, rxx)
                        excluded.append(i)
                    else:
                        c.gates[idx] = ZPhase(target=c.gates[idx].target, phase=phase)
                        commute_stack[g.target] = [('XPhase',i, g.phase, False)]
                else:
                    commute_stack[g.target].append(('XPhase',i, g.phase, False))

            elif g.name == 'ZPhase':
                if len(commute_stack[g.target]) > 0:
                    name, idx, phase, rxx = commute_stack[g.target][-1]
                    if name == 'HAD':
                        commute_stack[g.target].append(('ZPhase',i, g.phase, False)) #boolean indicates if RXX is in front
                    elif name == 'ZPhase':
                        # Merge ZPhases
                        commute_stack[g.target][-1] = (name, idx, phase + g.phase, rxx)
                        excluded.append(i)
                    else:
                        c.gates[idx] = XPhase(target=c.gates[idx].target, phase=phase)
                        commute_stack[g.target] = [('ZPhase',i, g.phase, False)]
                else:
                    commute_stack[g.target].append(('ZPhase',i, g.phase, False))

            elif g.name == 'RXX':
                # We allow XPhase gates commute through RXX gates
                if len(commute_stack[g.target]) > 0:
                    name, idx, phase, rxx = commute_stack[g.target][-1]
                    if name == 'HAD':
                        commute_stack[g.target] = []
                    elif name == 'ZPhase':
                        c.gates[idx] = ZPhase(target=c.gates[idx].target, phase=phase)
                        commute_stack[g.target] = []
                    elif not rxx:
                        commute_stack[g.target][-1] = (name, idx, phase, True)
                if len(commute_stack[g.control]) > 0:
                    name, idx, phase, rxx = commute_stack[g.control][-1]
                    if name == 'HAD':
                        commute_stack[g.control] = []
                    elif name == 'ZPhase':
                        c.gates[idx] = ZPhase(target=c.gates[idx].target, phase=phase)
                        commute_stack[g.control] = []
                    elif not rxx:
                        commute_stack[g.control][-1] = (name, idx, phase, True)
    for _,v in commute_stack.items():
        for name, idx, phase, _ in v:
            if name == 'XPhase':
                c.gates[idx] = XPhase(target=c.gates[idx].target, phase=phase)
            elif name == 'ZPhase':
                c.gates[idx] = ZPhase(target=c.gates[idx].target, phase=phase)

    c.gates = [g for i, g in enumerate(c.gates) if i not in excluded]

def apply_pending_gates(c: Circuit, gms_handler: Global_gate_handler):
    """
    Applies to input circuit the pending gates in status_qubits and current_global.
    """
    h_left = []
    rx = {}
    xx = []

    for ms in gms_handler.current_global:
        if ms.h_right:
            c.add_gate("HAD", ms.control)
        rx[ms.control] = rx.get(ms.control, 0) + 1
        rx[ms.target] = rx.get(ms.target, 0) + 1
        xx.append(RXX(ms.control, ms.target, Fraction(-1,2)))
        if ms.h_left:
            h_left.append(ms.control)

    for q, phase in rx.items():
        c.add_gate(XPhase(q, Fraction(phase, 2)))

    for gate in xx:
        c.add_gate(gate)

    for q in h_left:
        c.add_gate("HAD", q)
    
    for _, gate in gms_handler.status_qubits:
        c.add_gate(gate)

def apply_cnots(g: BaseGraph[VT, ET], c: Circuit, frontier: List[VT], qubit_map: Dict[VT, int],
                cnots: List[CNOT], m: Mat2, neighbors: List[VT], gms_handler: Global_gate_handler):
    """
    Adds the list of CNOTs to the circuit, modifying the graph, frontier, and qubit map as needed.
    Returns the number of vertices that end up being extracted
    """
    if len(cnots) > 0:
        cnots2 = cnots
        cnots = []
        for i, cnot in enumerate(cnots2):
            m.row_add(cnot.target, cnot.control)
            curr_cnot = CNOT(qubit_map[frontier[cnot.control]], qubit_map[frontier[cnot.target]])
            cnots.append(curr_cnot)
            gms_handler.q_tracker.setdefault(curr_cnot.control, []).append(i)
            gms_handler.q_tracker.setdefault(curr_cnot.target, []).append(i)
        connectivity_from_biadj(g, m, neighbors, frontier)
    
    good_verts = dict()
    for i, row in enumerate(m.data):
        if sum(row) == 1:
            v = frontier[i]
            w = neighbors[[j for j in range(len(row)) if row[j]][0]]
            good_verts[v] = w
    if not good_verts:
        raise Exception("No extractable vertex found. Something went wrong")
    outputs = g.outputs()
    hads = []
    good_qubits = []
    for v, w in good_verts.items():  # Update frontier vertices
        q = qubit_map[v]
        good_qubits.append(q)
        if q not in gms_handler.q_tracker: # Easy vertex
            if gms_handler.can_be_placed(q, 'HAD'):
                c.add_gate("HAD",q)
            else:
                gms_handler.status_qubits.append(({q},HAD(target=q)))
                gms_handler.busy_qubits.add(q)
        elif q == cnots[gms_handler.q_tracker[q][-1]].target:
            # These go after all the CNOTs in this layer
            hads.append(q)
        qubit_map[w] = qubit_map[v]
        b = [o for o in g.neighbors(v) if o in outputs][0]
        g.remove_vertex(v)
        g.add_edge((w, b))
        frontier.remove(v)
        frontier.append(w)
    
    for (q,gate) in gms_handler.status_qubits:
        gms_handler.busy_qubits = gms_handler.busy_qubits.union(q)

    for i, cnot in enumerate(cnots):
        h_left, h_right = gms_handler.classify_cnot(i, cnots, good_qubits)
        cnot = HCNOTH(cnot.control, cnot.target, h_left, h_right)
        if (gms_handler.fits_in_current_global(cnot)):
            gms_handler.add_cnot_to_current_global(cnot)
        else:
            # Place in the circuit pending SQGs and the CNOTs of the current global gate
            apply_pending_gates(c, gms_handler)
            gms_handler.update_global_gates()
            # Flush variables and track current cnot
            gms_handler.flush_variables()
            gms_handler.add_cnot_to_current_global(cnot)
    gms_handler.finish_layer(hads)
    return len(good_verts)

def clean_frontier(g: BaseGraph[VT, ET], c: Circuit, frontier: List[VT], qubit_map: Dict[VT, int],
                   gms_handler: Global_gate_handler, optimize_czs: bool = True) -> int:
    """Remove single qubit gates from the frontier and any CZs between the vertices in the frontier
    Returns the number of CZs saved if `optimize_czs` is True; otherwise returns 0
    """
    phases = g.phases()
    czs_saved = 0
    outputs = g.outputs()

    for v in frontier:  # First removing single qubit gates
        q = qubit_map[v]
        b = [w for w in g.neighbors(v) if w in outputs][0]
        e = g.edge(v, b)

        if g.edge_type(e) == EdgeType.HADAMARD:
            if gms_handler.can_be_placed(q, 'HAD'):
                c.add_gate("HAD",q)
            else:
                gms_handler.status_qubits.append( ({q},HAD(target=q)) )
                gms_handler.busy_qubits.add(q)
            g.set_edge_type(e, EdgeType.SIMPLE) # We pretend we placed the Hadamard already
        
        if phases[v]:
            if gms_handler.can_be_placed(q, 'ZPhase'):
                c.add_gate("ZPhase", q, phases[v])
            else:
                gms_handler.status_qubits.append( ({q}, ZPhase(target=q, phase=phases[v])))
                gms_handler.busy_qubits.add(q)
            g.set_phase(v, 0) # We pretend we placed the Rz already

    # And now on to CZ gates
    czs_now, czs_later = [], []
    czs_q_now, czs_q_later = {}, {}
    for v in frontier:
        for w in list(g.neighbors(v)):
            if w not in frontier:
                continue
            i = qubit_map[v]
            j = qubit_map[w]
            g.remove_edge(g.edge(v, w))
            if gms_handler.can_be_placed(i, 'CZ') and gms_handler.can_be_placed(j, 'CZ'):
                czs_q_now[i] = czs_q_now.get(i, 0) + 1
                czs_q_now[j] = czs_q_now.get(j, 0) + 1
                czs_now.append(RXX(i, j, Fraction(-1,2)))
            else:
                czs_q_later[i] = czs_q_later.get(i, 0) + 1
                czs_q_later[j] = czs_q_later.get(j, 0) + 1
                czs_later.append(({i,j},RXX(i, j, Fraction(-1,2))))

    if len(czs_now) != 0:
        gms_handler.cz_global_gates.append(czs_now[:])
        for q, count in czs_q_now.items():
            czs_now.append(XPhase(q, Fraction(count, 2)))
            czs_now.append(HAD(q))
            czs_now.insert(0, HAD(q))
        for g in reversed(czs_now):
            c.add_gate(g)

    if len(czs_later) != 0:
        gms_handler.cz_global_gates.append([g[1] for g in czs_later])
        for q, count in czs_q_later.items():
            czs_later.append(({q},XPhase(q, Fraction(count, 2))))
            czs_later.append(({q},HAD(q)))
            czs_later.insert(0, ({q},HAD(q)))
            gms_handler.busy_qubits.add(q)
        for g in czs_later:
            gms_handler.status_qubits.append(g)

    return czs_saved

def frontier_to_cnots(m2: Mat2):
    lp = create_LP(m2.data)
    lp['model'].verbose = 0
    lp['model'].optimize()
    cnots = []

    if lp['model'].status.value != 3 and lp['model'].status.value != 0:
        raise Exception

    for i in range(len(lp['gms'])):
        for j in range(len(lp['gms'][i])):
            if i != j and int(lp['gms'][i][j].x) == 1:
                cnots.append(CNOT(i,j))

    return cnots

def extract_peephole_only(g: BaseGraph[VT, ET], optimize_czs = False, optimize_cnots: int = 2,
        up_to_perm: bool = False, quiet: bool = True):
    gadgets = {}
    inputs = g.inputs()
    outputs = g.outputs()
    gms_handler = Global_gate_handler()

    c = Circuit(len(outputs))

    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in inputs and v not in outputs:
            n = list(g.neighbors(v))[0]
            gadgets[n] = v

    qubit_map: Dict[VT,int] = dict()
    frontier = []
    for i, o in enumerate(outputs):
        v = list(g.neighbors(o))[0]
        if v in inputs:
            continue
        frontier.append(v)
        qubit_map[v] = i

    czs_saved = 0
    
    while True:
        # Remove SQGs and CZs from frontier. status_qubits changed in-place.
        czs_saved += clean_frontier(g, c, frontier, qubit_map, gms_handler, optimize_czs)
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = neighbors_of_frontier(g, frontier)
        
        if not frontier:
            apply_pending_gates(c, gms_handler)
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if remove_gadget(g, frontier, qubit_map, neighbor_set, gadgets):
            # There was a gadget in the way. Go back to the top
            continue
            
        neighbors = list(neighbor_set)
        m = bi_adj(g, neighbors, frontier)
        if all(sum(row) != 1 for row in m.data):  # No easy vertex
            
            perm = column_optimal_swap(m)
            perm = {v: k for k, v in perm.items()}
            neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]
            m2 = bi_adj(g, neighbors2, frontier)
            if optimize_cnots > 0:
                cnots = m2.to_cnots(optimize=True)
            else:
                cnots = m2.to_cnots(optimize=False)

            # Since the matrix is not square, the algorithm sometimes introduces duplicates
            cnots = filter_duplicate_cnots(cnots)
            neighbors = neighbors2
            m = m2
        else:
            if not quiet: print("Simple vertex")
            cnots = []

        extracted = apply_cnots(g, c, frontier, qubit_map, cnots, m, neighbors, gms_handler)
        if not quiet: print("Vertices extracted:", extracted)
            
    # Outside of loop. Finish up the permutation
    id_simp(g, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates
    c.gates = list(reversed(c.gates))

    result_circuit = graph_to_swaps(g, up_to_perm) + c
    
    # We call it twice to simplify even further
    simplify_sqgs(result_circuit)
    simplify_sqgs(result_circuit)

    global_gates = gms_handler.global_gates
    cz_globals = gms_handler.cz_global_gates
    return result_circuit, global_gates, cz_globals

def extract_peephole_ILP(g: BaseGraph[VT, ET], optimize_czs = False,
        up_to_perm: bool = False, quiet: bool = True):
    gadgets = {}
    inputs = g.inputs()
    outputs = g.outputs()
    gms_handler = Global_gate_handler()

    c = Circuit(len(outputs))

    for v in g.vertices():
        if g.vertex_degree(v) == 1 and v not in inputs and v not in outputs:
            n = list(g.neighbors(v))[0]
            gadgets[n] = v

    qubit_map: Dict[VT,int] = dict()
    frontier = []
    for i, o in enumerate(outputs):
        v = list(g.neighbors(o))[0]
        if v in inputs:
            continue
        frontier.append(v)
        qubit_map[v] = i

    czs_saved = 0
    
    while True:
        # Remove SQGs and CZs from frontier. status_qubits changed in-place.
        czs_saved += clean_frontier(g, c, frontier, qubit_map, gms_handler, optimize_czs)
        
        # Now we can proceed with the actual extraction
        # First make sure that frontier is connected in correct way to inputs
        neighbor_set = neighbors_of_frontier(g, frontier)
        
        if not frontier:
            apply_pending_gates(c, gms_handler)
            break  # No more vertices to be processed. We are done.
        
        # First we check if there is a phase gadget in the way
        if remove_gadget(g, frontier, qubit_map, neighbor_set, gadgets):
            # There was a gadget in the way. Go back to the top
            continue
            
        neighbors = list(neighbor_set)
        m = bi_adj(g, neighbors, frontier)
        if all(sum(row) != 1 for row in m.data):  # No easy vertex
            
            perm = column_optimal_swap(m)
            perm = {v: k for k, v in perm.items()}
            neighbors2 = [neighbors[perm[i]] for i in range(len(neighbors))]
            m2 = bi_adj(g, neighbors2, frontier)

            cnots = frontier_to_cnots(m2)

            # Since the matrix is not square, the algorithm sometimes introduces duplicates
            cnots = filter_duplicate_cnots(cnots)
            neighbors = neighbors2
            m = m2
        else:
            if not quiet: print("Simple vertex")
            cnots = []

        extracted = apply_cnots(g, c, frontier, qubit_map, cnots, m, neighbors, gms_handler)
        if not quiet: print("Vertices extracted:", extracted)
            
    # Outside of loop. Finish up the permutation
    id_simp(g, quiet=True)  # Now the graph should only contain inputs and outputs
    # Since we were extracting from right to left, we reverse the order of the gates
    c.gates = list(reversed(c.gates))

    result_circuit = graph_to_swaps(g, up_to_perm) + c

    # We call it twice to simplify even further
    simplify_sqgs(result_circuit)
    simplify_sqgs(result_circuit)

    global_gates = gms_handler.global_gates
    cz_globals = gms_handler.cz_global_gates
    return result_circuit, global_gates, cz_globals

# == Below is the code to run the benchmarks == #

def timeout_handler(_, __):
    raise TimeoutError

def compile_circuits(in_path: str, out_path: str, extract_function: Callable):
    """
    Use `extract_function` to compile all the circuits in `in_path`.
    Save result in `out_path`.
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
        input_circuit, g = load_circuit(in_path, filename)
        if input_circuit == None:
            continue
        signal.alarm(900) # 15 min timeout
        start = time.time()
        try:
            result_circuit, global_gates, cz_globals = extract_function(g.copy())
        except TimeoutError:
            print("TO'd for", filename)
            with open(os.path.join(out_path, 'exec_times.csv'),"a") as fresult:
                fresult.write( ','.join([filename, "Timeout"] ) + '\n')
            continue
        end = time.time()
        signal.alarm(0)

        signal.alarm(300) # 5 min timeout
        start = time.time()
        try:
            equal = result_circuit.verify_equality(input_circuit)
            assert equal
        except TimeoutError:
            print("Equality check TO'd for", filename)
        end = time.time()
        signal.alarm(0)

        with open(os.path.join(out_path, filename),'w') as saved_circuit:
            saved_circuit.write(result_circuit.to_qasm())

        # Save how long we took to compile circuit
        result = {'filename': filename, 'time': end - start}
        
        with open(os.path.join(out_path, 'exec_times.csv'),"a") as fresult:
            fresult.write(','.join([str(x) for x in result.values()]) + '\n')

        # We write the circuit with a 'gms' qasm instruction instead of XX gates
        save_recompiled(out_path, filename, result_circuit)
        
        print("Finished for ", filename)

def run_benchmarks(path_mqt, path_qasmbench, path_chemistry, out_path):

    signal.signal(signal.SIGALRM, timeout_handler)
    
    # Step 1: remove measurements and barriers from input circuits
    remove_meas_and_barriers(path_mqt)
    remove_meas_and_barriers(path_qasmbench)
    remove_meas_and_barriers(path_chemistry)

    # Step 2: we run each method (peephole only, peephole + ILP, qiskit) in MQT, chemistry, and QASMBench
    ## MQT circuits
    compile_circuits(os.path.join(path_mqt, 'no_measurements'), os.path.join(out_path,'mqt','peephole'), extract_peephole_only)
    compile_circuits(os.path.join(path_mqt, 'no_measurements'), os.path.join(out_path,'mqt','ilp'), extract_peephole_ILP)
    compile_qiskit(os.path.join(path_mqt, 'no_measurements'), os.path.join(out_path,'mqt','qiskit'))
    ## QASMBench circuits
    compile_circuits(os.path.join(path_qasmbench, 'no_measurements'), os.path.join(out_path,'qbench','peephole'), extract_peephole_only)
    compile_circuits(os.path.join(path_qasmbench, 'no_measurements'), os.path.join(out_path,'qbench','ilp'), extract_peephole_ILP)
    compile_qiskit(os.path.join(path_qasmbench, 'no_measurements'), os.path.join(out_path,'qbench','qiskit'))
    ## Chemistry circuits
    compile_circuits(os.path.join(path_chemistry, 'no_measurements'), os.path.join(out_path,'chemistry','peephole'), extract_peephole_only)
    compile_circuits(os.path.join(path_chemistry, 'no_measurements'), os.path.join(out_path,'chemistry','ilp'), extract_peephole_ILP)
    compile_qiskit(os.path.join(path_chemistry, 'no_measurements'), os.path.join(out_path,'chemistry','qiskit'))

    # Step 3: we take the results and put them in a csv collecting the metrics we are intested in
    ## MQT
    original = os.path.join(path_mqt, 'no_measurements')
    ilp = os.path.join(out_path,'mqt','ilp', 'recompiled')
    peephole = os.path.join(out_path,'mqt','peephole', 'recompiled')
    qiskit = os.path.join(out_path,'mqt','qiskit')
    out = os.path.join(out_path,'mqt','mqt_comparisons.csv')
    save_time_comparisons(original, ilp, peephole, qiskit, out)
    ## QASMBench
    original = os.path.join(path_qasmbench, 'no_measurements')
    ilp = os.path.join(out_path,'qbench','ilp', 'recompiled')
    peephole = os.path.join(out_path,'qbench','peephole', 'recompiled')
    qiskit = os.path.join(out_path,'qbench','qiskit')
    out = os.path.join(out_path,'qbench','qbench_comparisons.csv')
    save_time_comparisons(original, ilp, peephole, qiskit, out)
    ## Chemistry
    original = os.path.join(path_chemistry, 'no_measurements')
    ilp = os.path.join(out_path,'chemistry','ilp', 'recompiled')
    peephole = os.path.join(out_path,'chemistry','peephole', 'recompiled')
    qiskit = os.path.join(out_path,'chemistry','qiskit')
    out = os.path.join(out_path,'chemistry','chemistry_comparisons.csv')
    save_time_comparisons(original, ilp, peephole, qiskit, out)

def run_second_naive_algorithm(path_mqt, path_qasmbench, path_chemistry, out_path):
    
    compile_second_naive(path_qasmbench, os.path.join(out_path, 'qbench', 'naive2'))
    compile_second_naive(path_mqt, os.path.join(out_path, 'mqt', 'naive2'))
    compile_second_naive(path_chemistry, os.path.join(out_path, 'chemistry', 'naive2'))
    
    save_time_second_naive(path_qasmbench, os.path.join(out_path, 'qbench', 'naive2'), os.path.join(out_path, 'qbench', 'qbench_comparisons.csv'))
    save_time_second_naive(path_mqt, os.path.join(out_path, 'mqt', 'naive2'), os.path.join(out_path, 'mqt', 'mqt_comparisons.csv'))
    save_time_second_naive(path_chemistry, os.path.join(out_path, 'chemistry', 'naive2'), os.path.join(out_path, 'chemistry', 'chemistry_comparisons.csv'))

if __name__ == '__main__':

    path_mqt = "path/to/mqt_circuits"
    path_chemistry = "path/to/chemistry_circuits"
    path_qasmbench = "path/to/qasmbench_circuits"
    out_path = "path/to/output"
    #To run the original benchmarks, uncomment line below. Change your paths to point to the quantum circuits accordingly.
    #run_benchmarks(path_mqt, path_qasmbench, path_chemistry, out_path)

    #Later we developed a second naive algorithm to compare with. Uncomment below to run.
    #run_second_naive_algorithm(path_mqt, path_qasmbench, path_chemistry, out_path)