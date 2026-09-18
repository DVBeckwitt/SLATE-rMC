# Automatic Codex discovery workflow

The personal `dream-rsi` skill is enabled by global Codex instructions. Codex consults
relevant experiment history automatically during substantial iterative work and uses
bounded search when the task has a repeatable evaluator and competing approaches.
The user does not need to name the skill. Ordinary direct edits remain direct work.
New local Codex sessions load the global instructions; an already-running session
can read the installed skill immediately. Other computers and cloud environments
need the same installation; this does not change an account-wide server setting.

## Ownership and installation

The maintained source is [tools/dream-rsi](../tools/dream-rsi/SKILL.md). It is development
tooling, never imported by `painted_ewald` or `rasim_next`, and adds no dependencies.
The installed copy lives in `$CODEX_HOME/skills/dream-rsi` (normally
`C:/Users/Kenpo/.codex/skills/dream-rsi`). The installer preserves existing global
instructions and updates only its marked block:

```powershell
python -B tools/dream-rsi/scripts/install.py
```

Global state is external under `$CODEX_HOME/dream-rsi`. Orchestration JSON, logs,
policy code and isolated candidate checkouts are development records, not scientific
trace artifacts. Scientific diagnostics still follow the single external `.ra_diag.npz`
rule. No state is written beneath this repository. The source remains versioned here;
rerun the installer after changing it. The global skill works in other Git repositories
without importing SLATE physics or applying SLATE's evaluator there.

## SLATE adapter

[configs/dream_rsi.json](../configs/dream_rsi.json) supplies a serial-writer profile,
protected proof/reference paths, software gates, external caches and finite budgets.
Before a search, Codex narrows the writable scope and adds any task-specific oracle,
numerical qualification and benchmark commands to an external copy of the profile.
The default score measures feasibility only; it does not rank performance or qualify
a physical fit. Missing scientific qualification remains a blocker for promotion.

```powershell
python -B tools/dream-rsi/scripts/dream_rsi.py history --repo .
python -B tools/dream-rsi/scripts/dream_rsi.py init --repo . `
  --task "The bounded, authorized task" --config C:/external/task-rsi.json
python -B tools/dream-rsi/scripts/dream_rsi.py cycle C:/external/returned-experiment
```

These are implementation interfaces for Codex, not steps the user must repeat.
The runner uses the invoking interpreter, so use SLATE's environment for its gates.
Its evaluation process sets `PYTHONPATH` to the candidate source rather than trusting
an editable installation that may point to the original checkout.

## Paper mapping and adaptation

[Dream-RSI, section 3](https://arxiv.org/html/2609.14858v1#S3) supplies the fixed-model,
fixed-evaluator architecture: online discovery trees, historical replay, executable
policy revision, and selection that includes the incumbent. The implementation uses
that section's quality-minus-work objective, with an optional parallelism bonus.
Appendix B motivates branch-history ranking, explicit repairs, saturation detection,
and a beta schedule fixed during an episode. No model weights are trained.

Our adaptations are dependency-respecting replay, an ancestry-only current-cycle
worker context by default, isolated local clones, a restricted Python policy subset,
strict file admission, hard budgets, and external append-only records. The API opens
one new root per batch, following section 3; it does not implement the appendix's
separate grid of multiple selectable unopened roots or its distinct Pareto-AUC score.
The authors' unreleased implementation is not a source-level dependency. Beta/default
heuristics are implementation choices, not universal optimum claims.

## Boundaries and limitations

- Every node retains parent, branch, depth, context dependencies, proposal, commit,
  patch, measured evaluation, failure class and execution cost. Failed evaluations
  have no score; the valid baseline remains eligible.
- A worker's changes are admitted by path and copied to a separate evaluator checkout.
  Protected comparator/reference changes are rejected. Workers still run ordinary
  repository code in Codex's sandbox; this is not a hostile-code security service.
- Replay invokes no coding agent or scientific evaluator. It reveals only recorded,
  dependency-eligible continuations. Missing continuations are unknown, not failures.
  Policies receive live-legal actions; unsupported choices censor a run rather than
  redirecting its policy. Selection compares the same complete incumbent replay cases.
  Prefix-only access does not prove that an alternative live ordering would generate
  the same programs. Old prose notes remain notes, not invented replay evidence.
- Policy code is archived before evaluation, limited to pure Python operations, and
  receives only observed numeric/status fields and legal actions. Every decision has
  a timeout. The evaluator, replay implementation, archive and scoring are not editable
  by the policy-development worker.
- Incumbents win replay ties. A policy's historical score is not a future guarantee.
  Feasibility-only searches with a passing baseline retain the fixed controller;
  there is no quality signal for learning a better policy. Repair tasks can explicitly
  admit failing pass-only baselines; the SLATE profile requires a valid baseline.
  No productivity study is required for deployment; the user explicitly waived it.
- A search starts from a clean committed state. It never resets, merges, or removes
  an unrelated checkout. Candidate adoption is a separate normal project integration.
- Time and output limits terminate only the owned process tree. Disk budget is checked
  between batches and may be exceeded within a running batch. Retained workspaces are
  not automatically deleted. Pending work is reported and explicitly recoverable.
- Seven permanent test groups cover replay visibility/dependencies, legal actions and
  restricted policies, correctness/repair behavior, the full offline cycle, protected
  paths/dirty inputs, timeout/recovery/global-install boundaries, and feasibility-only
  policy behavior. They use fake
  model responses and real isolated Git/evaluator processes, not paid model sweeps.

Scientific APIs, conventions, legacy classifications and numerical tolerances are
unchanged. This workflow does not close T35's numerical qualification gaps.
