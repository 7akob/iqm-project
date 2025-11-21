# 5-node Flow Optimization on Quantom Backend

## Overview
This experiment demonstrates solving a 5-node notwork flow optimization problem using IQM's and IonQ's quantom hardware through the `iqm-client` and `qiskit` integration. The goal is to find a optimal binary flow assignments that minimize energy(cost) while satysfying demand and capacity constraints for each node.

## How to setup
1. Rename `.env.example` to `.env` and set your personal api_token and quantom computer url in the `.env` file. 
2. Create a venv Python envoirement, **IMPORTANT for IQM use Python 3.11**
```bash
python3.11 -m venv project-env
source project-env/bin/activate
```
3. Install dependencies (inside envoirement)
```bash
pip install -r requirements_IQM.txt OR requirements_IONQ.txt
```
4. Run the script
```bash
cd ~/.../iqm-project/src
python run_5node_iqm.py OR run_5node_IONQ.py
```
### Example output from running on IQM backend (real quantom hardware) (using run_5node_iqm.py)
```
Problem uses 8 binary variables:
  ['f_A_C', 'f_A_D', 'f_A_E', 'f_B_C', 'f_B_D', 'f_B_E', 'f_E_C', 'f_E_D']
Built BQM: 8 linear terms; 18 quadratic terms
Connected to IQM backend: IQM Backend   qubits: 20
Starting optimization on IQM (this will run hardware multiple times).
params: [0.5, 0.5] -> energy: -127.8359
params: [1.5, 0.5] -> energy: -126.3301
params: [0.5, 1.5] -> energy: -130.3789
params: [-0.0095, 2.3605] -> energy: -110.0137
params: [0.9992, 1.5287] -> energy: -136.7676
params: [1.4894, 1.6272] -> energy: -139.6035
Optimization finished in 34.6 s; result:  message: Maximum number of function evaluations has been exceeded.
 success: False
  status: 2
     fun: -139.603515625
       x: [ 1.489e+00  1.627e+00]
    nfev: 6
   maxcv: 0.0

Final counts: {'10111101': 7, '10111100': 5, '10110001': 7, '11101100': 3, '00000111': 3, '10001010': 3, '00101001': 6, '01101010': 2, '10011101': 3, '10011001': 5, '10010100': 2, '10101011': 4, '01110111': 4, '01100100': 1, '10110111': 2, '10101001': 10, '11100011': 2, '01000011': 1, '00001011': 4, '10101100': 1, '10001100': 2, '10000111': 6, '11111001': 2, '01111011': 2, '11101111': 2, '00101000': 4, '00111110': 6, '11011000': 2, '11100110': 1, '11000101': 3, '10110101': 4, '10000101': 3, '11010101': 5, '11011101': 2, '10101000': 2, '10100011': 3, '11101001': 4, '11100001': 3, '01100010': 1, '10110011': 5, '10111001': 7, '01000110': 2, '10100001': 3, '10011000': 3, '00101111': 3, '10110100': 3, '00011101': 7, '10010101': 6, '01101011': 2, '00111001': 5, '01000101': 1, '10101101': 5, '10111000': 2, '11000010': 2, '00111010': 7, '00000000': 1, '00010101': 3, '01110101': 3, '00010111': 4, '00011111': 4, '01001111': 3, '00110100': 3, '00011011': 5, '01110001': 5, '11000001': 2, '00101101': 4, '00001101': 3, '11001110': 4, '00111101': 7, '10011111': 4, '01111000': 1, '11111000': 1, '10101111': 4, '00010001': 2, '10110000': 4, '00001001': 4, '11010111': 3, '01101110': 2, '11111011': 5, '11010011': 1, '10100111': 3, '11110011': 1, '00111011': 2, '10100000': 4, '10000100': 2, '11011001': 2, '01101111': 3, '01001000': 1, '01100001': 1, '11101101': 4, '11011010': 1, '00110000': 1, '00100011': 1, '10001000': 1, '00000110': 2, '11101011': 1, '00110001': 3, '11110100': 1, '11110101': 2, '11100111': 1, '00111000': 4, '01111111': 4, '11100101': 3, '10100100': 1, '00110111': 2, '00100110': 2, '01011111': 2, '01011100': 1, '10111110': 1, '10111011': 8, '00110110': 1, '01100110': 3, '01011110': 4, '01100111': 2, '01010111': 1, '10001101': 3, '10001111': 3, '10001001': 3, '01010010': 1, '10011100': 1, '00000010': 1, '10000110': 1, '10100101': 5, '10000000': 3, '10000010': 1, '00110011': 6, '01001101': 2, '01010101': 2, '00100000': 2, '11011110': 2, '10011011': 2, '00011110': 2, '01001010': 3, '00000001': 2, '00011010': 2, '00011001': 2, '10010001': 2, '10011010': 4, '00010010': 2, '00100101': 3, '00100111': 1, '01011001': 2, '01111001': 3, '11101110': 2, '11100100': 1, '11111010': 1, '00101011': 5, '11011011': 1, '01101001': 2, '00010100': 1, '11001011': 1, '01001001': 2, '00110101': 1, '11001101': 1, '10000001': 2, '00100001': 2, '11010100': 2, '10010010': 2, '01110010': 1, '00111111': 2, '11000011': 1, '01110000': 1, '00001000': 1, '11110010': 2, '10111010': 2, '10110010': 2, '01100011': 3, '01111010': 1, '00000101': 2, '01111101': 1, '01101101': 2, '11110111': 1, '10111111': 3, '01001110': 2, '11101000': 1, '10010000': 4, '00000011': 1, '10010111': 2, '01111110': 1, '01011000': 1, '10011110': 1, '11101010': 1, '10100010': 1, '01011101': 2, '11111100': 1, '01110011': 2, '11110110': 1, '00001111': 1, '00010000': 2, '01111100': 2, '11110000': 1, '00011100': 1, '11111110': 1, '00001110': 1, '00110010': 2, '11111101': 1, '10101010': 2, '01110100': 3, '11111111': 1, '01010000': 1, '00101100': 2}

Best measured bitstring: 10101001
Energy: -153.0

Active flows (1 == flow on arc):
  A -> C
  A -> E
  B -> D
  E -> D

Demand checks:
  C: 1/3 ✗
  D: 2/2 ✓

Source caps:
  A: 2/3 ✓
  B: 1/2 ✓

```

### Example output from runnin on IONQ simulated backend (using run_5node_ionq.py)
```
Problem: 8 binary variables
Variables: ['f_A_C', 'f_A_D', 'f_A_E', 'f_B_C', 'f_B_D', 'f_B_E', 'f_E_C', 'f_E_D']

Built BQM: 8 linear, 12 quadratic terms

/home/jakob/Code/iqm-project/src/ionq-env/lib64/python3.14/site-packages/qiskit_ionq/ionq_backend.py:127: IonQTranspileLevelWarning: Transpiler default optimization_level=2. IonQ (QIS) recommends 0-1 to avoid aggressive re-synthesis; use transpile(..., optimization_level=1).
  warn_bad_transpile_level()
Connected to: ionq_simulator

Starting QAOA optimization on IonQ...
(Each iteration runs a quantum circuit on hardware)

  params: [0.5, 0.5] -> energy: -156.2402
  params: [1.5, 0.5] -> energy: -143.7988
  params: [0.5, 1.5] -> energy: -140.748
  params: [-0.1262, -0.2797] -> energy: -137.4766
  params: [0.3284, 0.9696] -> energy: -124.6211
  params: [0.4574, 0.2537] -> energy: -139.2246

Optimization complete!
Best parameters: [np.float64(0.5), np.float64(0.5)]

Running final circuit with optimized parameters...
Final counts: {'00000001': 1, '00001010': 1, '00001011': 2, '00001100': 2, '00001110': 2, '00010010': 2, '00011000': 1, '00011010': 6, '00011110': 1, '00100000': 2, '00100010': 2, '00100011': 2, '00100100': 1, '00100110': 1, '00100111': 1, '00101000': 2, '00101001': 2, '00101010': 6, '00101011': 2, '00101100': 4, '00101110': 11, '00110000': 1, '00110001': 1, '00110010': 6, '00110011': 3, '00110100': 2, '00110110': 2, '00111000': 4, '00111001': 4, '00111010': 26, '00111011': 6, '00111100': 7, '00111110': 18, '01000010': 1, '01100010': 2, '01100110': 2, '01101010': 1, '01101100': 1, '01101110': 2, '01110010': 5, '01110110': 1, '01111000': 1, '01111010': 2, '01111110': 5, '01111111': 1, '10000000': 1, '10000010': 4, '10000101': 1, '10000110': 1, '10000111': 1, '10001000': 4, '10001001': 2, '10001010': 8, '10001011': 1, '10001100': 1, '10001110': 5, '10010000': 1, '10010001': 1, '10010010': 1, '10010100': 1, '10010101': 1, '10010110': 1, '10011000': 2, '10011001': 1, '10011010': 5, '10011011': 1, '10011100': 1, '10011101': 1, '10011110': 1, '10100001': 1, '10100010': 1, '10100011': 2, '10100100': 3, '10100110': 2, '10100111': 2, '10101000': 6, '10101001': 3, '10101010': 20, '10101011': 2, '10101100': 9, '10101101': 1, '10101110': 19, '10110000': 1, '10110001': 3, '10110010': 7, '10110011': 2, '10110100': 4, '10110101': 3, '10110110': 2, '10111000': 23, '10111001': 9, '10111010': 49, '10111011': 4, '10111100': 40, '10111101': 1, '10111110': 39, '11000010': 2, '11000100': 1, '11000110': 1, '11001010': 3, '11010010': 1, '11010100': 2, '11011010': 1, '11011110': 1, '11100000': 1, '11100001': 1, '11100010': 4, '11101010': 1, '11101100': 1, '11101110': 1, '11110000': 1, '11110001': 3, '11110010': 5, '11110011': 1, '11110100': 6, '11110110': 3, '11111000': 2, '11111010': 3, '11111100': 1, '11111110': 2, '11111111': 1}

======================================================================
SOLUTION
======================================================================

Best bitstring: 10111010
Energy: -182.00

Active flows:
  A → C
  A → E
  B → C
  B → D
  E → C

Total cost: 10.00

Demand checks:
  ✓ C: 3/3
  ✗ D: 1/2

Capacity checks:
  ✓ A: 2/3
  ✓ B: 2/2

======================================================================

```


## Problem Setup
- Nodes A, B, C, D, E
- Binary variables (8 total): represent possible flows between node pairs
```python
['f_A_C', 'f_A_D', 'f_A_E', 'f_B_C', 'f_B_D', 'f_B_E', 'f_E_C', 'f_E_D']
```
- Objective: minimize the total energy of the Binary Quadratic Model (BQM)
- Constraints:
  - Source capacity limits A and B
  - Demand requirements at C and D

## Quantom backend
- Backend: IQM Quantom Processor (I used IQM Garnet)
- Qubits available: 20
- IQM Client version 32.1.1
- Execution: 6 function evalutaions using real quantom hardware

## Results
- Optimization status: Finished (Max function evaluations reached)
- Best parameters: `[1.489, 1.627]`
- Minimum energy found: `-139.6035`
- Best mesurement bitstring: `10101001`

### Decoded active flows
- A → C
- A → E
- B → D
- E → D

### Constraint check
| Node | Required/Capacity | Actual | Status |
|------|-------------------|--------|--------|
| C    | 3                 | 1      | x      |
| D    | 2                 | 2      | ✓      |
| A    | <3                | 2      | ✓      |
| B    | <2                | 1      | ✓      |


## Interpretation
The quantom optimizer successfully found a near-optimal flow configuration with minimal violations:
- Node D's demand fully satisfied
- Node C's under supplied (1/3)
- All source capacity constraints respected
- Indicates strong performance despite limited iteration count and hardware noise
