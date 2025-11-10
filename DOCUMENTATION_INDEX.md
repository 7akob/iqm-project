# Documentation Index

## Overview
This project demonstrates solving a 5-node network flow optimization problem using IQM quantum computing tools and QAOA (Quantum Approximate Optimization Algorithm).

## Quick Access

### For First-Time Users
1. **Start here**: [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
2. **Understand the network**: [NETWORK_DIAGRAM.md](NETWORK_DIAGRAM.md) - Visual topology
3. **Deep dive**: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Complete tutorial

### For Developers
1. **Example code**: `src/example_network_optimization.py` - Annotated demonstration
2. **IQM implementation**: `src/toy_5node_qubo.py` - Production code with quantum backend
3. **Requirements**: `requirements.txt` - All dependencies

## Document Descriptions

### [README.md](README.md)
- Project overview and features
- Installation instructions
- Usage examples
- Test results from local and IQM executions
- Project structure

### [QUICKSTART.md](QUICKSTART.md) (5 min read)
- Minimal setup instructions
- Quick run guide
- Expected results
- Basic troubleshooting

### [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) (30 min read)
- Comprehensive step-by-step tutorial
- Detailed problem description and parameters
- QUBO formulation mathematics
- Constraint handling with penalty methods
- IQM backend configuration
- QAOA setup and execution
- Result interpretation and validation
- Parameter tuning guidelines
- Advanced troubleshooting

### [NETWORK_DIAGRAM.md](NETWORK_DIAGRAM.md) (5 min read)
- ASCII art network topology
- Flow path descriptions
- Constraint visualization
- Example solutions
- Decision variables explanation
- Network properties table

## Code Files

### `src/example_network_optimization.py`
**Purpose**: Educational demonstration with detailed logging

**Features**:
- Step-by-step execution with clear output
- Explains each phase (parameters, variables, QUBO, solving, interpretation)
- Uses classical ExactSolver for demonstration
- Validates constraints and shows violations
- Can run without IQM access

**When to use**: Learning how the optimization works, testing problem formulations

**Run**: `python src/example_network_optimization.py`

### `src/toy_5node_qubo.py`
**Purpose**: Production implementation for IQM quantum computers

**Features**:
- Connects to IQM quantum backend
- Implements QAOA with quantum sampler
- Uses actual quantum hardware
- Requires IQM credentials

**When to use**: Running on actual quantum hardware, production optimization

**Run**: `python src/toy_5node_qubo.py` (requires `.env` with IQM credentials)

## Learning Path

### Beginner
1. Read README.md overview
2. Follow QUICKSTART.md to run the example
3. Review NETWORK_DIAGRAM.md to understand the topology
4. Run `src/example_network_optimization.py` to see detailed output

### Intermediate
1. Study OPTIMIZATION_GUIDE.md sections 1-3 (problem formulation)
2. Understand QUBO construction in OPTIMIZATION_GUIDE.md sections 3-4
3. Examine `src/example_network_optimization.py` code
4. Experiment with different penalty weights

### Advanced
1. Complete OPTIMIZATION_GUIDE.md including QAOA and tuning sections
2. Set up IQM access and configure `.env`
3. Study `src/toy_5node_qubo.py` quantum implementation
4. Run on IQM hardware and compare with classical results
5. Extend to larger networks or different constraints

## Problem Summary

**Objective**: Minimize power generation costs in a network

**Network Components**:
- 2 Sources (A, B): Generators with costs and capacity limits
- 2 Sinks (C, D): Consumers with specific demands
- 1 Battery (E): Storage with 50% initial charge

**Approach**:
1. Formulate as QUBO (Quadratic Unconstrained Binary Optimization)
2. Encode constraints using penalty methods
3. Solve using QAOA on IQM quantum computer
4. Interpret and validate binary solution

**Solution Space**: 2^8 = 256 possible flow configurations

**Key Insight**: This demonstrates how quantum computing can solve combinatorial optimization problems relevant to energy grid management, logistics, and resource allocation.

## Getting Help

**Installation issues**: See QUICKSTART.md troubleshooting section

**IQM connection problems**: See OPTIMIZATION_GUIDE.md Step 4.1 and troubleshooting

**Constraint violations**: See OPTIMIZATION_GUIDE.md Step 8 (tuning penalty weights)

**Understanding QUBO**: See OPTIMIZATION_GUIDE.md Step 2-3

**Network topology questions**: See NETWORK_DIAGRAM.md

## Related Topics

This project demonstrates concepts applicable to:
- **Energy Management**: Power grid optimization, battery scheduling
- **Supply Chain**: Flow optimization, inventory management
- **Transportation**: Route planning, resource allocation
- **Telecommunications**: Network flow, bandwidth allocation
- **Quantum Computing**: QAOA, QUBO formulation, constraint encoding

## Contributing

To extend this project:
1. Modify network parameters in the Python files
2. Add new constraints to the QUBO formulation
3. Test different penalty weights and QAOA configurations
4. Scale to larger networks (10+ nodes)
5. Implement time-varying demands or costs
6. Compare quantum vs. classical solver performance

## References

- **IQM Client**: https://iqm-finland.github.io/iqm-client/
- **Qiskit Optimization**: https://qiskit.org/documentation/optimization/
- **QUBO Formulation**: https://docs.dwavesys.com/docs/latest/
- **QAOA Tutorial**: https://qiskit.org/textbook/ch-applications/qaoa.html
