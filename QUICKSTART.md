# Quick Start: 5-Node Network Optimization

This quick start guide gets you running the 5-node network optimization problem in minutes.

## Problem Overview

**Goal**: Minimize generation costs while satisfying power demands in a 5-node network.

**Network**:
- Sources: A (max 3 units @ $2/unit), B (max 2 units @ $3/unit)
- Sinks: C (needs 3 units), D (needs 2 units)
- Battery: E (4 units capacity, starts at 50%)

**Visual**: See [NETWORK_DIAGRAM.md](NETWORK_DIAGRAM.md) for network topology diagram.

## Quick Setup

### 1. Install Dependencies

```bash
python -m venv iqm-env
source iqm-env/bin/activate  # Windows: iqm-env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure IQM Access

Create `.env` file:
```bash
SERVER_URL=your_iqm_server_url
RESONANCE_API_TOKEN=your_api_token
```

### 3. Run the Optimization

```bash
python src/toy_5node_qubo.py
```

## Expected Results

Optimal solution:
- Flow paths: A→C, A→D, A→E, B→C, B→E, E→C
- Total cost: $12.0
- All demands satisfied ✓
- Battery balanced at 50% ✓

## What's Happening?

1. **QUBO Formulation**: The network flow problem is converted to a binary optimization problem
2. **IQM Connection**: Connects to IQM quantum computer
3. **QAOA Solver**: Uses Quantum Approximate Optimization Algorithm
4. **Result Interpretation**: Extracts and validates the flow solution

## Next Steps

For detailed explanation of each step, see [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)

## Troubleshooting

**Connection failed?**
- Check credentials in `.env`
- Verify API token is valid

**Constraints violated?**
- Increase penalty weight (line 49 in toy_5node_qubo.py)
- Try classical solver first for testing

**Need help?**
- See full guide: [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)
- Check IQM docs: https://iqm-finland.github.io/iqm-client/
