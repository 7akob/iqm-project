import itertools
import numpy as np
import dimod

# Problem data
sources = ['A', 'B']
sinks = {'C': 3, 'D': 2}
battery = 'E'
initial_soc = 0.5
battery_capacity = 4

Gmax = {'A': 3, 'B': 2}
cost = {'A': 2.0, 'B': 3.0}

nodes = ['A', 'B', 'C', 'D', 'E']
arcs = [(i, j) for i in nodes for j in nodes if i != j]

valid_arcs = [
    (i, j)
    for (i, j) in arcs
    if (i in sources and j in sinks.keys())
    or (i in sources and j == battery)
    or (i == battery and j in sinks.keys())
]

var_names = [f"f_{i}_{j}" for (i, j) in valid_arcs]

# Build QUBO
linear = {}
quadratic = {}
penalty = 8.0

# Source costs
for v in var_names:
    parts = v.split("_")
    src = parts[1]
    if src in sources:
        linear[v] = linear.get(v, 0.0) + cost[src]

# Demand satisfaction
for sink, demand in sinks.items():
    incoming = [v for v in var_names if v.endswith("_" + sink)]

    for v in incoming:
        linear[v] = linear.get(v, 0.0) + penalty * 1.0
        linear[v] = linear.get(v, 0.0) - 2.0 * penalty * demand
    for (v1, v2) in itertools.combinations(incoming, 2):
        quadratic_key = tuple(sorted([v1, v2]))
        quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2.0 * penalty

# Source generation limits
for src in sources:
    outgoing = [v for v in var_names if v.startswith(f"f_{src}_")]
    gmax = Gmax[src]
    for v in outgoing:
        linear[v] = linear.get(v, 0.0) + penalty * 1.0
        linear[v] = linear.get(v, 0.0) - 2.0 * penalty * gmax
    for (v1, v2) in itertools.combinations(outgoing, 2):
        quadratic_key = tuple(sorted([v1, v2]))
        quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2.0 * penalty

# Battery SOC balance
incoming_to_bat = [v for v in var_names if v.endswith("_E")]
outgoing_from_bat = [v for v in var_names if v.startswith("f_E_")]

target_soc = 0.5 * battery_capacity

for v in incoming_to_bat + outgoing_from_bat:
    coeff = 1.0 if v in incoming_to_bat else -1.0
    linear[v] = linear.get(v, 0.0) + penalty * (coeff**2)
    linear[v] = linear.get(v, 0.0) - 2.0 * penalty * target_soc * coeff

for (v1, v2) in itertools.combinations(incoming_to_bat + outgoing_from_bat, 2):
    coeff1 = 1.0 if v1 in incoming_to_bat else -1.0
    coeff2 = 1.0 if v2 in incoming_to_bat else -1.0
    quadratic_key = tuple(sorted([v1, v2]))
    quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2.0 * penalty * (coeff1 * coeff2)

# Build and sample BQM
bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
for v, c in linear.items():
    bqm.add_variable(v, c)
for (v1, v2), q in quadratic.items():
    bqm.add_interaction(v1, v2, q)

sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=200)
best = sampleset.first

# Results
print("Best sample (binary values):")
print(best.sample)
print("Energy:", best.energy)

print("\nInterpreting flows (active arcs = 1 means 1 unit flow):")
for v in var_names:
    if best.sample[v] == 1:
        print(" ", v)

def inflow_to(node, sample):
    return sum(sample[v] for v in var_names if v.endswith("_" + node))

for sink in sinks:
    print(f"Sink {sink} demand {sinks[sink]}, received {inflow_to(sink, best.sample)}")
