import os
import signal
import time
import qiskit.qasm2
from qiskit import transpile
from natsort import natsorted

def compile_qiskit(in_path: str, out_path: str):
    """
    Benchmark qiskit transpiler on the circuits in `in_path`. Save result in `out_path`.
    Qiskit converts the input circuit into one that uses XX gates and single-qubit rotations.
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
            try:
                qc = qiskit.qasm2.load(os.path.join(in_path, filename), custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            except Exception:
                print('Qiskit failed loading', filename)
                continue
            signal.alarm(900) # 15 min timeout
            start = time.time()
            try:
                result_qc = transpile(qc, basis_gates=['rx','rz','rxx'], optimization_level=3)
            except TimeoutError:
                print("TO'd for", filename)
                fresult.write( ','.join([filename, "Timeout"] ) + '\n')
                continue
            end = time.time()
            signal.alarm(0)

            fresult.write(';'.join([filename, str(end - start)]) + '\n')

            with open(os.path.join(out_path, filename),'w') as saved_circuit:
                saved_circuit.write(qiskit.qasm2.dumps(result_qc))

            print("Finished for", filename)