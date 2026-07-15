# Execution-plan convention

Long tasks update the execution-plan section in their assigned task file under `tasks/`. Do not create an additional planning document unless the task explicitly requests one.

Use these states:

```text
NS         not started
PLANNED    source audit, equations, contracts, proof cases, and stop conditions recorded
ORACLE     analytic or independent oracle passes
REF        public reference implementation passes analytic checks
LEGACY     shared characterization-pack comparisons are classified
CONVERGED  numerical refinement converges
BENCH      wall time and peak memory recorded
READY      all merge gates pass
BLOCKED    smallest unmet shared requirement recorded
```

At each transition, record:

```text
State:
Evidence:
Commands run:
Remaining work:
Contract or dependency issue:
```

A plan must identify the exact owned paths, forbidden paths, manuscript equations, original-RASIM source locations, shared-pack cases, independent proof, convergence variable, benchmark workload, and done conditions before source editing begins.
