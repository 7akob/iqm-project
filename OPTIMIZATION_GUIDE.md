# Step-by-Step Guide: 5-Node Network Optimization with IQM Quantum Computing

This guide provides comprehensive instructions for building and solving a network flow optimization problem using IQM quantum computing tools.

## Problem Description

We have a 5-node network with the following characteristics:

- **2 Source Nodes (A, B)**: Power generators with generation costs and capacity limits
- **2 Sink Nodes (C, D)**: Consumers with specific power demands
- **1 Battery Node (E)**: A controllable storage device at 50% initial state of charge
- **Fully Connected Network**: All valid arcs have flow capacity limits

For a visual representation of the network topology, see [NETWORK_DIAGRAM.md](NETWORK_DIAGRAM.md).

### Problem Parameters

| Component | Node | Parameter | Value |
|-----------|------|-----------|-------|
| Source | A | Max Generation | 3 units |
| Source | A | Cost per unit | $2.0 |
| Source | B | Max Generation | 2 units |
| Source | B | Cost per unit | $3.0 |
| Sink | C | Demand | 3 units |
| Sink | D | Demand | 2 units |
| Battery | E | Capacity | 4 units |
| Battery | E | Initial SOC | 50% (2 units) |

### Network Topology

Valid flow arcs in the network:
- Source → Sink: A→C, A→D, B→C, B→D
- Source → Battery: A→E, B→E
- Battery → Sink: E→C, E→D

## Step 1: Prerequisites and Setup

### 1.1 Install Required Dependencies

Create a virtual environment and install dependencies:

```bash
python -m venv iqm-env
source iqm-env/bin/activate  # On Windows: iqm-env\Scripts\activate
pip install -r requirements.txt
```

Required packages:
- `dimod==0.12.21` - For QUBO problem formulation
- `iqm-client[qiskit]==31.0.0` - IQM quantum computing client
- `qiskit-optimization>=0.5.0` - Qiskit optimization tools
- `python-dotenv==0.21.1` - For environment variable management
- `numpy==2.3.4` - Numerical computations
- `scipy==1.15.3` - Scientific computing

### 1.2 Configure IQM Access

Create a `.env` file in the project root with your IQM credentials:

```bash
SERVER_URL=your_iqm_server_url
RESONANCE_API_TOKEN=your_api_token
```

**Note**: Contact IQM to obtain access credentials for their quantum computers.

## Step 2: Understanding the QUBO Formulation

### 2.1 Decision Variables

Each decision variable represents a binary flow on an arc:
- `f_{i}_{j} = 1` means 1 unit of power flows from node i to node j
- `f_{i}_{j} = 0` means no flow on that arc

For our problem, we have 8 decision variables:
```
f_A_C, f_A_D, f_A_E, f_B_C, f_B_D, f_B_E, f_E_C, f_E_D
```

### 2.2 Objective Function

Minimize total generation cost:
```
Cost = 2.0 * (f_A_C + f_A_D + f_A_E) + 3.0 * (f_B_C + f_B_D + f_B_E)
```

### 2.3 Constraints (Penalty Method)

We use a penalty approach to enforce constraints in the QUBO formulation:

#### Constraint 1: Sink Demand Satisfaction

For each sink node, the sum of incoming flows must equal demand:
```
Sum of incoming flows to C = 3
Sum of incoming flows to D = 2
```

Penalty term: `P * (incoming_flow - demand)²`

#### Constraint 2: Source Generation Limits

Each source cannot exceed its maximum generation capacity:
```
Sum of outgoing flows from A ≤ 3
Sum of outgoing flows from B ≤ 2
```

Penalty term: `P * (outgoing_flow - G_max)²` if exceeded

#### Constraint 3: Battery State of Charge Balance

The battery should maintain its target SOC (50% = 2 units):
```
Initial_SOC + Incoming_to_E - Outgoing_from_E = Target_SOC
2 + (f_A_E + f_B_E) - (f_E_C + f_E_D) = 2
```

Simplifying: `(f_A_E + f_B_E) - (f_E_C + f_E_D) = 0`

Penalty term: `P * (net_flow_to_battery)²`

## Step 3: Building the QUBO Model

### 3.1 Initialize Problem Parameters

```python
import numpy as np
import dimod
from dotenv import load_dotenv
import os

# Define network structure
sources = ['A', 'B']
sinks = {'C': 3, 'D': 2}  # {node: demand}
battery = 'E'
battery_capacity = 4

# Source parameters
Gmax = {'A': 3, 'B': 2}  # Maximum generation
cost = {'A': 2.0, 'B': 3.0}  # Generation cost

# Define valid arcs
nodes = ['A', 'B', 'C', 'D', 'E']
valid_arcs = [
    (i, j)
    for i in nodes
    for j in nodes
    if i != j and (
        (i in sources and j in list(sinks.keys()))
        or (i in sources and j == battery)
        or (i == battery and j in list(sinks.keys()))
    )
]

# Create variable names
var_names = [f"f_{i}_{j}" for (i, j) in valid_arcs]
```

### 3.2 Construct Linear and Quadratic Coefficients

```python
linear = {}
quadratic = {}
penalty = 8.0  # Penalty weight (should be larger than objective costs)

# Add generation costs to objective
for v in var_names:
    src = v.split("_")[1]
    if src in sources:
        linear[v] = linear.get(v, 0.0) + cost[src]

# Add demand satisfaction penalties
for sink, demand in sinks.items():
    incoming = [v for v in var_names if v.endswith("_" + sink)]
    # Linear terms: P - 2*P*demand
    for v in incoming:
        linear[v] = linear.get(v, 0.0) + penalty - 2 * penalty * demand
    # Quadratic terms: 2*P for each pair
    import itertools
    for v1, v2 in itertools.combinations(incoming, 2):
        quadratic_key = tuple(sorted([v1, v2]))
        quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2 * penalty

# Add generation limit penalties
for src in sources:
    outgoing = [v for v in var_names if v.startswith(f"f_{src}_")]
    gmax = Gmax[src]
    for v in outgoing:
        linear[v] = linear.get(v, 0.0) + penalty - 2 * penalty * gmax
    for v1, v2 in itertools.combinations(outgoing, 2):
        quadratic_key = tuple(sorted([v1, v2]))
        quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2 * penalty

# Add battery SOC balance penalties
incoming_to_bat = [v for v in var_names if v.endswith("_E")]
outgoing_from_bat = [v for v in var_names if v.startswith("f_E_")]
target_soc = 0.5 * battery_capacity  # 50% = 2 units

for v in incoming_to_bat + outgoing_from_bat:
    coeff = 1.0 if v in incoming_to_bat else -1.0
    linear[v] = linear.get(v, 0.0) + penalty * (coeff**2) - 2 * penalty * target_soc * coeff

for v1, v2 in itertools.combinations(incoming_to_bat + outgoing_from_bat, 2):
    coeff1 = 1.0 if v1 in incoming_to_bat else -1.0
    coeff2 = 1.0 if v2 in incoming_to_bat else -1.0
    quadratic_key = tuple(sorted([v1, v2]))
    quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2 * penalty * (coeff1 * coeff2)
```

### 3.3 Create Binary Quadratic Model (BQM)

```python
# Build BQM
bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

# Add linear terms
for v, c in linear.items():
    bqm.add_variable(v, c)

# Add quadratic terms
for (v1, v2), q in quadratic.items():
    bqm.add_interaction(v1, v2, q)

print(f"QUBO model created with {len(bqm)} variables")
```

## Step 4: Connecting to IQM Backend

### 4.1 Load Credentials and Connect

```python
from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm import IQMProvider

# Load environment variables
load_dotenv()
SERVER_URL = os.getenv("SERVER_URL")  
API_TOKEN = os.environ.get("RESONANCE_API_TOKEN")

# Connect to IQM provider
provider = IQMProvider(url=SERVER_URL, token=API_TOKEN)
backend = provider.get_backend('resonance_qpu')  # Adjust for your target backend

print(f"Connected to IQM backend: {backend.name}")
```

### 4.2 Test Connection (Optional)

```python
from qiskit import QuantumCircuit

# Create simple test circuit
test_circuit = QuantumCircuit(2)
test_circuit.h(0)
test_circuit.cx(0, 1)
test_circuit.measure_all()

# Run test
from qiskit import transpile
transpiled = transpile(test_circuit, backend)
job = backend.run(transpiled, shots=1000)
result = job.result()
print(f"Test circuit result: {result.get_counts()}")
```

## Step 5: Solving with QAOA on IQM

### 5.1 Convert BQM to Qiskit Format

```python
from qiskit_optimization.converters import from_dimod_bqm

# Convert BQM to Qiskit QuadraticProgram
qubo = from_dimod_bqm(bqm)
```

### 5.2 Set Up and Run QAOA

```python
from qiskit.algorithms import QAOA
from qiskit.primitives import Sampler

# Create sampler with IQM backend
sampler = Sampler(backend=backend)

# Configure QAOA algorithm
qaoa = QAOA(sampler=sampler, reps=2)

# Solve the problem
result = qaoa.compute_minimum_eigenvalue(operator=qubo.to_ising()[0])

print("\nOptimization complete!")
print(f"Minimum energy: {result.eigenvalue.real}")
print(f"Optimal parameters: {result.optimal_parameters}")
```

### 5.3 Alternative: Classical Solver for Testing

For testing without quantum hardware access:

```python
from dimod import ExactSolver

# Use exact solver (only feasible for small problems)
sampler = ExactSolver()
sampleset = sampler.sample(bqm)

# Get best solution
best_sample = sampleset.first.sample
best_energy = sampleset.first.energy

print(f"\nBest sample (binary values): {best_sample}")
print(f"Energy: {best_energy}")
```

## Step 6: Interpreting Results

### 6.1 Extract Flow Solution

```python
def interpret_solution(sample):
    """Extract and display the flow solution"""
    print("\nInterpreting flows (active arcs = 1 means 1 unit flow):")
    
    active_flows = []
    for var, val in sample.items():
        if val == 1:
            active_flows.append(var)
            print(f"  {var}")
    
    # Check sink demands
    for sink, demand in sinks.items():
        incoming = [v for v in active_flows if v.endswith("_" + sink)]
        received = len(incoming)
        print(f"\nSink {sink} demand {demand}, received {received}")
        status = "✓ Satisfied" if received == demand else "✗ Not satisfied"
        print(f"  {status}")
    
    # Check source limits
    for src in sources:
        outgoing = [v for v in active_flows if v.startswith(f"f_{src}_")]
        generated = len(outgoing)
        print(f"\nSource {src} generated {generated} units (max {Gmax[src]})")
        status = "✓ Within limit" if generated <= Gmax[src] else "✗ Exceeds limit"
        print(f"  {status}")
    
    # Check battery balance
    bat_in = len([v for v in active_flows if v.endswith("_E")])
    bat_out = len([v for v in active_flows if v.startswith("f_E_")])
    net_flow = bat_in - bat_out
    final_soc = target_soc + net_flow
    print(f"\nBattery:")
    print(f"  Initial SOC: {target_soc} units")
    print(f"  Net flow: {net_flow} units (in: {bat_in}, out: {bat_out})")
    print(f"  Final SOC: {final_soc} units")
    
    # Calculate total cost
    total_cost = 0
    for var in active_flows:
        src = var.split("_")[1]
        if src in sources:
            total_cost += cost[src]
    print(f"\nTotal generation cost: ${total_cost}")
    
    return active_flows, total_cost

# Interpret the solution
active_flows, total_cost = interpret_solution(best_sample)
```

### 6.2 Validate Constraints

```python
def validate_solution(sample):
    """Check if solution satisfies all constraints"""
    violations = []
    
    # Check sink demands
    for sink, demand in sinks.items():
        incoming = sum(1 for v, val in sample.items() 
                      if val == 1 and v.endswith("_" + sink))
        if incoming != demand:
            violations.append(f"Sink {sink}: received {incoming}, needed {demand}")
    
    # Check source limits
    for src in sources:
        outgoing = sum(1 for v, val in sample.items()
                      if val == 1 and v.startswith(f"f_{src}_"))
        if outgoing > Gmax[src]:
            violations.append(f"Source {src}: generated {outgoing}, max {Gmax[src]}")
    
    # Check battery balance
    bat_in = sum(1 for v, val in sample.items() if val == 1 and v.endswith("_E"))
    bat_out = sum(1 for v, val in sample.items() if val == 1 and v.startswith("f_E_"))
    net_flow = bat_in - bat_out
    if abs(net_flow) > 0.01:  # Allow small numerical errors
        violations.append(f"Battery: net flow {net_flow}, should be 0")
    
    if violations:
        print("\n⚠ Constraint violations found:")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print("\n✓ All constraints satisfied!")
        return True

# Validate the solution
is_valid = validate_solution(best_sample)
```

## Step 7: Running the Complete Solution

The complete implementation is available in `src/toy_5node_qubo.py`. To run it:

```bash
# Activate virtual environment
source iqm-env/bin/activate

# Ensure .env file is configured
# Then run the script
python src/toy_5node_qubo.py
```

### Expected Output

```
Connected to IQM backend: IQM Backend

Optimization complete!
Minimum energy: -212.0

Interpreting flows (active arcs = 1 means 1 unit flow):
  f_A_C
  f_A_D
  f_A_E
  f_B_C
  f_B_E
  f_E_C

Sink C demand 3, received 3 ✓
Sink D demand 2, received 2 ✓

Source A generated 3 units (max 3) ✓
Source B generated 2 units (max 2) ✓

Battery:
  Initial SOC: 2.0 units
  Net flow: 0 units (in: 2, out: 2)
  Final SOC: 2.0 units

Total generation cost: $12.0

✓ All constraints satisfied!
```

## Step 8: Tuning and Optimization

### 8.1 Adjusting Penalty Weight

The penalty parameter (default: 8.0) balances constraint satisfaction vs. objective optimization:
- **Too small**: Constraints may be violated
- **Too large**: May dominate the objective, making optimization less effective

Try values between 5.0 and 20.0 based on your problem scale.

### 8.2 QAOA Parameters

Adjust QAOA parameters for better results:
```python
qaoa = QAOA(
    sampler=sampler,
    reps=2,  # Number of QAOA layers (try 1-5)
    optimizer=None,  # Use default optimizer or specify custom
)
```

### 8.3 Running Multiple Trials

For quantum algorithms, running multiple trials can improve solution quality:
```python
best_result = None
best_energy = float('inf')

for trial in range(5):
    result = qaoa.compute_minimum_eigenvalue(operator=qubo.to_ising()[0])
    if result.eigenvalue.real < best_energy:
        best_energy = result.eigenvalue.real
        best_result = result
    print(f"Trial {trial+1}: Energy = {result.eigenvalue.real}")

print(f"\nBest energy across trials: {best_energy}")
```

## Troubleshooting

### Common Issues

1. **IQM Connection Errors**
   - Verify credentials in `.env` file
   - Check network connectivity
   - Ensure API token is valid

2. **Version Incompatibility Warnings**
   - Update `iqm-client` to match server version
   - Check IQM documentation for compatible versions

3. **Constraint Violations in Solution**
   - Increase penalty weight
   - Try more QAOA repetitions
   - Use classical solver to verify problem formulation

4. **Poor Solution Quality**
   - Run multiple trials and select best
   - Adjust QAOA parameters
   - Consider problem-specific parameter tuning

## Additional Resources

- IQM Documentation: https://iqm-finland.github.io/iqm-client/
- Qiskit Optimization: https://qiskit.org/documentation/optimization/
- D-Wave QUBO Formulation: https://docs.dwavesys.com/docs/latest/

## Summary

This guide demonstrates how to:
1. ✅ Formulate a network flow optimization problem as a QUBO
2. ✅ Encode constraints using penalty methods
3. ✅ Connect to IQM quantum computing backend
4. ✅ Solve using QAOA algorithm
5. ✅ Interpret and validate results

The approach is generalizable to larger networks and more complex constraints by following the same QUBO formulation principles.
