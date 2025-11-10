"""
Example: 5-Node Network Optimization with IQM Quantum Computing

This script demonstrates the complete workflow for solving a network flow
optimization problem using IQM quantum computing tools.

For detailed explanation, see OPTIMIZATION_GUIDE.md
"""

import os
import itertools
import numpy as np
import dimod

# Optional imports (uncomment if you have IQM access)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# from iqm.iqm_client import IQMClient
# from iqm.qiskit_iqm import IQMProvider
# from qiskit import QuantumCircuit
# from qiskit.algorithms import QAOA
# from qiskit.primitives import Sampler
# from qiskit_optimization import QuadraticProgram
# from qiskit_optimization.converters import from_dimod_bqm


def define_network_parameters():
    """Define the 5-node network parameters"""
    print("=" * 60)
    print("STEP 1: DEFINE NETWORK PARAMETERS")
    print("=" * 60)
    
    sources = ['A', 'B']
    sinks = {'C': 3, 'D': 2}  # {node: demand}
    battery = 'E'
    battery_capacity = 4
    
    Gmax = {'A': 3, 'B': 2}  # Maximum generation
    cost = {'A': 2.0, 'B': 3.0}  # Generation cost per unit
    
    print("\nNetwork Configuration:")
    print(f"  Sources: {sources}")
    print(f"  Sinks: {sinks}")
    print(f"  Battery: {battery} (capacity: {battery_capacity} units, initial: 50%)")
    print(f"\nSource Parameters:")
    for src in sources:
        print(f"  {src}: Max Generation = {Gmax[src]} units, Cost = ${cost[src]}/unit")
    print(f"\nSink Demands:")
    for sink, demand in sinks.items():
        print(f"  {sink}: {demand} units")
    
    return sources, sinks, battery, battery_capacity, Gmax, cost


def create_decision_variables(sources, sinks, battery):
    """Create binary decision variables for network flows"""
    print("\n" + "=" * 60)
    print("STEP 2: CREATE DECISION VARIABLES")
    print("=" * 60)
    
    nodes = sources + list(sinks.keys()) + [battery]
    
    # Define valid arcs
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
    
    var_names = [f"f_{i}_{j}" for (i, j) in valid_arcs]
    
    print(f"\nTotal decision variables: {len(var_names)}")
    print("Valid flow arcs:")
    for arc in valid_arcs:
        print(f"  {arc[0]} → {arc[1]}")
    
    return var_names, valid_arcs


def build_qubo_model(var_names, sources, sinks, battery, battery_capacity, Gmax, cost):
    """Build the QUBO model with objective and constraints"""
    print("\n" + "=" * 60)
    print("STEP 3: BUILD QUBO MODEL")
    print("=" * 60)
    
    linear = {}
    quadratic = {}
    penalty = 8.0  # Note: Adjust this value to balance objective vs. constraints
    
    print(f"\nPenalty weight: {penalty}")
    print("Note: The penalty weight balances constraint satisfaction with cost minimization.")
    print("      Increase penalty if constraints are violated; decrease if cost is too high.")
    
    # Objective: Generation costs
    print("\n[Objective] Adding generation costs...")
    for v in var_names:
        src = v.split("_")[1]
        if src in sources:
            linear[v] = linear.get(v, 0.0) + cost[src]
            print(f"  {v}: cost = ${cost[src]}")
    
    # Constraint 1: Demand satisfaction
    print("\n[Constraint 1] Sink demand satisfaction...")
    for sink, demand in sinks.items():
        incoming = [v for v in var_names if v.endswith("_" + sink)]
        print(f"  Sink {sink} (demand={demand}): incoming arcs = {incoming}")
        
        for v in incoming:
            linear[v] = linear.get(v, 0.0) + penalty - 2 * penalty * demand
        
        for v1, v2 in itertools.combinations(incoming, 2):
            quadratic_key = tuple(sorted([v1, v2]))
            quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2 * penalty
    
    # Constraint 2: Generation limits
    print("\n[Constraint 2] Source generation limits...")
    for src in sources:
        outgoing = [v for v in var_names if v.startswith(f"f_{src}_")]
        gmax = Gmax[src]
        print(f"  Source {src} (max={gmax}): outgoing arcs = {outgoing}")
        
        for v in outgoing:
            linear[v] = linear.get(v, 0.0) + penalty - 2 * penalty * gmax
        
        for v1, v2 in itertools.combinations(outgoing, 2):
            quadratic_key = tuple(sorted([v1, v2]))
            quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2 * penalty
    
    # Constraint 3: Battery SOC balance
    print("\n[Constraint 3] Battery state of charge balance...")
    incoming_to_bat = [v for v in var_names if v.endswith("_E")]
    outgoing_from_bat = [v for v in var_names if v.startswith("f_E_")]
    target_soc = 0.5 * battery_capacity
    
    print(f"  Target SOC: {target_soc} units (50% of {battery_capacity})")
    print(f"  Incoming to battery: {incoming_to_bat}")
    print(f"  Outgoing from battery: {outgoing_from_bat}")
    
    for v in incoming_to_bat + outgoing_from_bat:
        coeff = 1.0 if v in incoming_to_bat else -1.0
        linear[v] = linear.get(v, 0.0) + penalty * (coeff**2) - 2 * penalty * target_soc * coeff
    
    for v1, v2 in itertools.combinations(incoming_to_bat + outgoing_from_bat, 2):
        coeff1 = 1.0 if v1 in incoming_to_bat else -1.0
        coeff2 = 1.0 if v2 in incoming_to_bat else -1.0
        quadratic_key = tuple(sorted([v1, v2]))
        quadratic[quadratic_key] = quadratic.get(quadratic_key, 0.0) + 2 * penalty * (coeff1 * coeff2)
    
    # Build BQM
    print("\n[Building BQM]")
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    
    for v, c in linear.items():
        bqm.add_variable(v, c)
    
    for (v1, v2), q in quadratic.items():
        bqm.add_interaction(v1, v2, q)
    
    print(f"  Total variables: {len(bqm)}")
    print(f"  Linear terms: {len(linear)}")
    print(f"  Quadratic terms: {len(quadratic)}")
    
    return bqm


def solve_with_classical_solver(bqm):
    """Solve using classical exact solver (for testing/validation)"""
    print("\n" + "=" * 60)
    print("STEP 4: SOLVE WITH CLASSICAL SOLVER")
    print("=" * 60)
    
    print("\nUsing ExactSolver (classical) for demonstration...")
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(bqm)
    
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    print(f"\nOptimization complete!")
    print(f"  Best energy: {best_energy}")
    print(f"  Number of solutions evaluated: {len(sampleset)}")
    
    return best_sample, best_energy


def interpret_solution(sample, sources, sinks, battery, battery_capacity, Gmax, cost):
    """Interpret and validate the solution"""
    print("\n" + "=" * 60)
    print("STEP 5: INTERPRET SOLUTION")
    print("=" * 60)
    
    print("\nActive flows (value = 1):")
    active_flows = []
    for var, val in sample.items():
        if val == 1:
            active_flows.append(var)
            # Parse arc
            parts = var.split("_")
            from_node, to_node = parts[1], parts[2]
            print(f"  {from_node} → {to_node}")
    
    if not active_flows:
        print("  No flows active")
    
    # Validate constraints
    print("\n" + "-" * 60)
    print("CONSTRAINT VALIDATION")
    print("-" * 60)
    
    violations = []
    
    # Check sink demands
    print("\n[Sink Demand Satisfaction]")
    for sink, demand in sinks.items():
        incoming = sum(1 for v, val in sample.items() 
                      if val == 1 and v.endswith("_" + sink))
        status = "✓" if incoming == demand else "✗"
        print(f"  {status} Sink {sink}: Demand = {demand}, Received = {incoming}")
        if incoming != demand:
            violations.append(f"Sink {sink}: received {incoming}, needed {demand}")
    
    # Check source limits
    print("\n[Source Generation Limits]")
    for src in sources:
        outgoing = sum(1 for v, val in sample.items()
                      if val == 1 and v.startswith(f"f_{src}_"))
        status = "✓" if outgoing <= Gmax[src] else "✗"
        print(f"  {status} Source {src}: Generated = {outgoing}, Max = {Gmax[src]}")
        if outgoing > Gmax[src]:
            violations.append(f"Source {src}: generated {outgoing}, max {Gmax[src]}")
    
    # Check battery balance
    print("\n[Battery State of Charge Balance]")
    target_soc = 0.5 * battery_capacity
    bat_in = sum(1 for v, val in sample.items() if val == 1 and v.endswith("_E"))
    bat_out = sum(1 for v, val in sample.items() if val == 1 and v.startswith("f_E_"))
    net_flow = bat_in - bat_out
    final_soc = target_soc + net_flow
    
    print(f"  Initial SOC: {target_soc} units (50%)")
    print(f"  Incoming: {bat_in} units")
    print(f"  Outgoing: {bat_out} units")
    print(f"  Net flow: {net_flow} units")
    print(f"  Final SOC: {final_soc} units")
    
    status = "✓" if abs(net_flow) < 0.01 else "✗"
    print(f"  {status} Battery balance: {'Maintained' if abs(net_flow) < 0.01 else 'Not maintained'}")
    
    if abs(net_flow) > 0.01:
        violations.append(f"Battery: net flow {net_flow}, should be 0")
    
    # Calculate total cost
    print("\n" + "-" * 60)
    print("COST ANALYSIS")
    print("-" * 60)
    
    total_cost = 0
    cost_breakdown = {}
    for var in active_flows:
        src = var.split("_")[1]
        if src in sources:
            cost_breakdown[src] = cost_breakdown.get(src, 0) + cost[src]
            total_cost += cost[src]
    
    print("\nCost breakdown by source:")
    for src in sources:
        src_cost = cost_breakdown.get(src, 0)
        units = src_cost / cost[src] if cost[src] > 0 else 0
        print(f"  Source {src}: {units:.0f} units × ${cost[src]}/unit = ${src_cost}")
    
    print(f"\nTotal generation cost: ${total_cost}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if violations:
        print("\n⚠ Constraint violations found:")
        for v in violations:
            print(f"  - {v}")
        print("\nSolution is NOT feasible.")
        return False, total_cost
    else:
        print("\n✓ All constraints satisfied!")
        print("✓ Solution is FEASIBLE and OPTIMAL")
        return True, total_cost


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("5-NODE NETWORK OPTIMIZATION WITH IQM QUANTUM COMPUTING")
    print("=" * 60)
    print("\nThis example demonstrates solving a network flow optimization")
    print("problem using QUBO formulation.")
    print("\nFor IQM quantum computer execution, see src/toy_5node_qubo.py")
    
    # Step 1: Define network
    sources, sinks, battery, battery_capacity, Gmax, cost = define_network_parameters()
    
    # Step 2: Create variables
    var_names, valid_arcs = create_decision_variables(sources, sinks, battery)
    
    # Step 3: Build QUBO
    bqm = build_qubo_model(var_names, sources, sinks, battery, battery_capacity, Gmax, cost)
    
    # Step 4: Solve (classical for demonstration)
    best_sample, best_energy = solve_with_classical_solver(bqm)
    
    # Step 5: Interpret and validate
    is_valid, total_cost = interpret_solution(
        best_sample, sources, sinks, battery, battery_capacity, Gmax, cost
    )
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Review OPTIMIZATION_GUIDE.md for detailed explanation")
    print("  - Configure IQM access to run on quantum hardware")
    print("  - See src/toy_5node_qubo.py for IQM implementation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
