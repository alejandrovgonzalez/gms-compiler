This is the implementation of the algorithm presented in the paper _Optimization and Synthesis of Quantum Circuits with Global Gates_.
The files are organized as follows:
* `synthesis_algorithms.py` implements the GMS compilation algorithm, with and without the linear program for the CNOT extraction part. This file also contains the main program to run the benchmarks.
* `qiskit_synthesis.py` implements the compilation with Qiskit that we compare against.
* `benchmark_utils.py` contains a variety of helper functions that are used for the compilation and benchmarking.
* `extract_LP.py` contains the implementation of the linear program.
* The folder `circuits/` contains the circuits we have used as input for the benchmarks as well as the benchmarking results.

Questions about the code or the paper can be addressed to <a.d.villoria.gonzalez@liacs.leidenuniv.nl>.