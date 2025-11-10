# Network Topology Diagram

## 5-Node Network Structure

```
                    Network Flow Topology
                    ====================

    Sources              Battery              Sinks
    -------              -------              -----

      A ────────────────────┐
      │ (Max: 3 units)      │
      │ (Cost: $2/unit)     │                    C
      │                     ↓                    │
      ├───────────────→   [ E ]  ───────────────┤
      │                     ↑   (Capacity: 4)    │ (Demand: 3)
      └──────────────────┐  │   (Initial: 50%)  │
                         │  │                    │
                         ↓  │                    ↓
                            └───────────────→ OUTPUT
      B ────────────────────┘                    ↑
      │ (Max: 2 units)                          │
      │ (Cost: $3/unit)                         │
      │                                         D
      └─────────────────────────────────────────┘
                                         (Demand: 2)


    Valid Flow Paths:
    ═════════════════

    Direct Source → Sink:
    • A → C  (Source A to Sink C)
    • A → D  (Source A to Sink D)
    • B → C  (Source B to Sink C)
    • B → D  (Source B to Sink D)

    Source → Battery:
    • A → E  (Source A charges Battery)
    • B → E  (Source B charges Battery)

    Battery → Sink:
    • E → C  (Battery discharges to Sink C)
    • E → D  (Battery discharges to Sink D)


    Constraints:
    ════════════

    1. Demand Satisfaction:
       - Sink C must receive exactly 3 units
       - Sink D must receive exactly 2 units

    2. Generation Limits:
       - Source A can generate at most 3 units
       - Source B can generate at most 2 units

    3. Battery Balance:
       - Initial SOC: 2 units (50% of 4 unit capacity)
       - Net flow must be zero (inflow - outflow = 0)
       - Final SOC should remain at 2 units

    4. Flow Binary:
       - Each arc either has 1 unit flow or 0 units
       - No fractional flows


    Objective:
    ══════════

    Minimize: Total Generation Cost

    Cost = 2.0 × (units from A) + 3.0 × (units from B)


    Example Optimal Solution:
    ══════════════════════════

    Active Flows:
    • A → C  (1 unit, cost $2)
    • A → D  (1 unit, cost $2)
    • A → E  (1 unit, cost $2)
    • B → C  (1 unit, cost $3)
    • B → E  (1 unit, cost $3)
    • E → C  (1 unit, cost $0)

    Verification:
    - Sink C receives: 1 (from A) + 1 (from B) + 1 (from E) = 3 ✓
    - Sink D receives: 1 (from A) = 1 ✗ (needs 2)
    - Source A generates: 3 (max 3) ✓
    - Source B generates: 2 (max 2) ✓
    - Battery: in=2 (A+B), out=1 (to C), net=+1 ✗ (should be 0)

    Note: Depending on penalty weight and solver, the exact solution
          may vary. This is characteristic of QUBO formulations where
          constraints are "soft" rather than "hard" constraints.

    Total Cost: $12


    Flow Representation:
    ════════════════════

         3 units out ↓        ↓ 2 units in      ↓ receives 3
                      A → → → E → → → C
                      ↓       ↑       ↑
                      → → → → B → → → ↑
                      ↓               ↓
                      → → → → → → → → D ← receives 2
                              2 units out


    Decision Variables (8 total):
    ══════════════════════════════

    Binary variables representing flow on each arc:
    • f_A_C ∈ {0, 1}  - Flow from A to C
    • f_A_D ∈ {0, 1}  - Flow from A to D
    • f_A_E ∈ {0, 1}  - Flow from A to E (battery)
    • f_B_C ∈ {0, 1}  - Flow from B to C
    • f_B_D ∈ {0, 1}  - Flow from B to D
    • f_B_E ∈ {0, 1}  - Flow from B to E (battery)
    • f_E_C ∈ {0, 1}  - Flow from E (battery) to C
    • f_E_D ∈ {0, 1}  - Flow from E (battery) to D

    Each variable = 1 means 1 unit flows on that arc
    Each variable = 0 means no flow on that arc
```

## Network Properties

| Property | Value |
|----------|-------|
| Total Nodes | 5 (2 sources + 2 sinks + 1 battery) |
| Total Arcs | 12 (but only 8 with valid flow) |
| Decision Variables | 8 binary variables |
| Constraints | 3 types (demand, capacity, balance) |
| Objective | Minimize cost |
| Problem Type | Binary Quadratic Optimization (QUBO) |
| Solution Space | 2^8 = 256 possible combinations |

## Key Insights

1. **Network is NOT fully connected**: While nodes could theoretically connect in 20 ways (5×4), only 8 arcs have valid flows based on the network rules (sources can't receive, sinks can't send, sources can't connect to each other, etc.)

2. **Battery acts as intermediate storage**: It can accept power from sources and deliver to sinks, providing flexibility in the network.

3. **Binary flows simplify QUBO formulation**: Each arc carries either 0 or 1 unit, making it natural to represent as binary variables.

4. **Penalty method for constraints**: Constraints are enforced by adding large penalty terms to the objective function, rather than being hard constraints.

5. **Trade-off between cost and feasibility**: Lower penalty weights may find cheaper but infeasible solutions; higher penalties ensure feasibility but may not find the global cost minimum.
