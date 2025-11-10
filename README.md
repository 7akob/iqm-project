# IQM Project: Network Flow Optimization with Quantum Computing

This project demonstrates how to solve a network flow optimization problem using IQM quantum computing tools and the QAOA (Quantum Approximate Optimization Algorithm).

## 📚 Documentation

**New users start here**: See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for a complete guide to all documentation.

## 🚀 Quick Start

**New to this project?** Start here:
- [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Complete step-by-step tutorial

## Problem Description

We solve a **5-node network optimization problem** with:
- **2 Source Nodes** (A, B): Power generators with costs and capacity limits
- **2 Sink Nodes** (C, D): Consumers with specific demands
- **1 Battery Node** (E): Controllable storage at 50% initial charge

**Objective**: Minimize generation costs while satisfying all demands and respecting network constraints.

## Features

✅ QUBO (Quadratic Unconstrained Binary Optimization) formulation  
✅ Integration with IQM quantum computers  
✅ QAOA algorithm implementation  
✅ Comprehensive constraint handling (demands, limits, battery balance)  
✅ Solution validation and interpretation  

## Installation

```bash
# Clone the repository
git clone https://github.com/7akob/iqm-project.git
cd iqm-project

# Create virtual environment
python -m venv iqm-env
source iqm-env/bin/activate  # Windows: iqm-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your IQM credentials:
```bash
SERVER_URL=your_iqm_server_url
RESONANCE_API_TOKEN=your_api_token
```

## Usage

Run the optimization:
```bash
python src/toy_5node_qubo.py
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick 5-minute setup and run guide
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Complete tutorial with:
  - Detailed problem formulation
  - Step-by-step QUBO construction
  - IQM backend setup
  - QAOA configuration
  - Result interpretation
  - Troubleshooting tips
- **[NETWORK_DIAGRAM.md](NETWORK_DIAGRAM.md)** - Visual network topology and flow paths

## Example Results

### Optimal Solution
```
Connected to IQM backend: IQM Backend

Optimization complete!
Minimum energy: -212.0

Active flows:
  f_A_C, f_A_D, f_A_E, f_B_C, f_B_E, f_E_C

✓ Sink C demand 3, received 3
✓ Sink D demand 2, received 2
✓ Source A generated 3 units (max 3)
✓ Source B generated 2 units (max 2)
✓ Battery balanced at 50%

Total generation cost: $12.0
✓ All constraints satisfied!
```

## Test Results

### Local Testing
```
Best sample (binary values): {'f_A_C': np.int8(1), 'f_A_D': np.int8(1), 'f_A_E': np.int8(1), 'f_B_C': np.int8(1), 'f_B_D': np.int8(0), 'f_B_E': np.int8(1), 'f_E_C': np.int8(1), 'f_E_D': np.int8(0)} 
Energy: -212.0
Interpreting flows (active arcs = 1 means 1 unit flow): f_A_C f_A_D f_A_E f_B_C f_B_E f_E_C 
Sink C demand 3, received 3 
Sink D demand 2, received 1
```

### IQM Quantum Computer
```
UserWarning: Your IQM Client version 31.0.0 was built for a different version of IQM Server. 
You might encounter issues. For the best experience, consider using a version of IQM Client 
that satisfies 32.1.1 <= iqm-client < 33.0.
  warnings.warn(version_incompatibility_msg)
Connected to IQM backend: IQM Backend
Test circuit result: {'11': 484, '10': 29, '00': 470, '01': 17}
Best sample (binary values):
{'f_A_C': np.int8(1), 'f_A_D': np.int8(1), 'f_A_E': np.int8(1), 'f_B_C': np.int8(1), 'f_B_D': np.int8(1), 'f_B_E': np.int8(0), 'f_E_C': np.int8(0), 'f_E_D': np.int8(0)}
Energy: -212.0

IQM result: [ { "11": 484, "00": 470 } ]
```

## Project Structure

```
iqm-project/
├── src/
│   └── toy_5node_qubo.py    # Main optimization implementation
├── QUICKSTART.md            # Quick start guide
├── OPTIMIZATION_GUIDE.md    # Comprehensive tutorial
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Dependencies

- `dimod==0.12.21` - QUBO/BQM formulation
- `iqm-client[qiskit]==31.0.0` - IQM quantum computing
- `qiskit-optimization>=0.5.0` - Optimization tools
- `python-dotenv==0.21.1` - Environment variables
- `numpy==2.3.4` - Numerical computing
- `scipy==1.15.3` - Scientific computing

## Troubleshooting

See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md#troubleshooting) for common issues and solutions.

## Resources

- [IQM Documentation](https://iqm-finland.github.io/iqm-client/)
- [Qiskit Optimization](https://qiskit.org/documentation/optimization/)
- [QUBO Formulation Guide](https://docs.dwavesys.com/docs/latest/)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]