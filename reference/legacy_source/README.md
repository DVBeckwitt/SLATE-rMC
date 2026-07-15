# Read-only original-RASIM source snapshot

This directory contains only the original files cited by the task documents. It exists so a clean
worktree can inspect the exact legacy equations, parameter semantics, and intermediate behavior
without mounting another repository.

It is evidence, not architecture. Never import it from `src/`, copy modules wholesale, add it to
`sys.path`, or make permanent tests depend on executing it. Reimplement the mathematics behind the
new contracts. Use `reference/rasim_reference_v1.npz` for immutable numerical comparisons and the
tracked manuscript extracts for the selected physical equations.

Original path references in task documents are relative to this directory. The source archive hash
and every tracked file hash are recorded in `MANIFEST.json`. The original license is included.
