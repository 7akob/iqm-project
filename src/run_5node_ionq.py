import os
import numpy as np
from dotenv import load_dotenv

# Qiskit core imports
from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_ionq import IonQProvider
from qiskit.primitives import Sampler

# Load credentials
load_dotenv()
IONQ_API_KEY = os.getenv("IONQ_API_KEY")

if not IONQ_API_KEY:
    raise EnvironmentError("Missing IONQ_API_KEY in .env file")

provider = IonQProvider(token=IONQ_API_KEY)
backend = provider.get_backend("ionq_qpu")  # or "ionq_simulator"
print(f"✅ Connected to IonQ backend: {backend.name()}")

# Define network flow problem
nodes = ["A", "B", "C", "D", "E"]

sources = {"A": {"cap": 3, "cost": 1}, "B": {"cap": 2, "cost": 1}}
sinks = {"C": {"demand": 3}, "D": {"demand": 2}}
storage = {"E": {"cap": 2, "init": 1}}

arcs = [
    ("A", "C"), ("A", "D"), ("A", "E"),
    ("B", "C"), ("B", "D"), ("B", "E"),
    ("E", "C"), ("E", "D")
]

var_names = [f"f_{i}_{j}" for (i, j) in arcs]
print(f"\nProblem uses {len(arcs)} binary variables:")
print(" ", var_names)

# Build QUBO model
qubo = QuadraticProgram()

for name in var_names:
    qubo.binary_var(name)

# Linear term = source generation costs
linear = {f"f_{i}_{j}": sources.get(i, {}).get("cost", 0) for (i, j) in arcs}
quadratic = {}

# Add penalties for unmet demand
for sink, data in sinks.items():
    related = [f"f_{i}_{sink}" for (i, j) in arcs if j == sink]
    for a in related:
        quadratic[(a, a)] = quadratic.get((a, a), 0) + 1
        linear[a] -= 2 * data["demand"]

qubo.minimize(linear=linear, quadratic=quadratic)

# Convert to Ising form (modern Qiskit version)
ising = qubo.to_ising()

if len(ising) == 2:
    operator, offset = ising
    print(f"\nBuilt Ising Hamiltonian with offset {offset:.3f}")
else:
    linear, quadratic, offset = ising
    print(f"\nBuilt BQM: {len(linear)} linear terms; {len(quadratic)} quadratic terms")

# Setup and run QAOA on IonQ
sampler = Sampler()
qaoa = QAOA(sampler=sampler, reps=2, optimizer=COBYLA(maxiter=50))
optimizer = MinimumEigenOptimizer(qaoa)

print("\n🚀 Running QAOA optimization on IonQ (this may take several minutes)...")
result = optimizer.solve(qubo)

# Display results
bitstring = "".join(str(int(v)) for v in result.x)
print("\n=== Optimization Result ===")
print(f"Best measured bitstring: {bitstring}")
print(f"Objective value: {result.fval:.3f}")

print("\nActive flows (1 = flow on arc):")
for k, val in zip(var_names, result.x):
    if val > 0.5:
        i, j = k.split("_")[1:]
        print(f"  {i} -> {j}")

# Sanity checks
print("\nDemand checks:")
for sink, data in sinks.items():
    inflow = sum(result.variables_dict.get(f"f_{i}_{sink}", 0) for i in sources.keys() | storage.keys())
    print(f"  {sink}: {inflow}/{data['demand']} {'✓' if inflow >= data['demand'] else '✗'}")

print("\nSource capacity checks:")
for src, data in sources.items():
    outflow = sum(result.variables_dict.get(f"f_{src}_{j}", 0) for j in nodes if (src, j) in arcs)
    print(f"  {src}: {outflow}/{data['cap']} {'✓' if outflow <= data['cap'] else '✗'}")

print("\n✅ Finished successfully.")
