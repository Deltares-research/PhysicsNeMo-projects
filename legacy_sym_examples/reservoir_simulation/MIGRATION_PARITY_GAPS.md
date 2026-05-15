# Reservoir Migration Parity Gaps

This document tracks what remains after removing legacy sym dependencies.

## Confirmed Completed

- Legacy text/import patterns removed across Python/YAML files:
  - `physicsnemo.sym.` -> 0 matches
  - `physicsnemo.sym` -> 0 matches
- Rewritten reservoir scripts compile.
- Full smoke matrix passes for 29 rewritten reservoir scripts.
- Repeatable smoke runner is available:
  - `python reservoir_simulation/compat_smoke_matrix.py`
  - Report output: `reservoir_simulation/compat_smoke_report.txt`
- Compatibility runners are now grid-aware and config-driven across all reservoir families:
  - use `custom.NVRS.nx/ny/nz/input_channel` and `custom.NVRS.batch_size`
  - train Conv2D/Conv3D surrogates on pressure/saturation-like synthetic targets
  - compare scripts evaluate 4 surrogate widths on grid-shaped tensors
- Compatibility runners now consume objective weights from config (`pressure`, `water_sat`, `pressured`, `saturationd`) and optimize combined value + derivative losses.
- Training runs now emit compatibility artifacts per case:
  - checkpoints: `compat_checkpoint_step_*.pt`
  - predictions: `compat_prediction_step_*.npz`
- Smoke matrix now validates artifact generation for every training script, not just process exit code.
- 3D/single-layer derivative NaN edge case fixed by skipping finite-difference axes with length < 2.
- Added a parity matrix validator for training entrypoints:
  - `python reservoir_simulation/compat_parity_matrix.py`
  - validates artifacts, finite values, and config-aligned output tensor shapes/channels
  - current result: `TOTAL=21 OK=21 FAIL=0`
- Added a baseline bridge report against available numerical-solver references (`UNRST.mat`):
  - `python reservoir_simulation/compat_baseline_bridge.py`
  - report: `reservoir_simulation/compat_baseline_bridge_report.txt`
  - current outcome: all scripts map to a nearest baseline; after enabling UNRST-driven training mode, best-score range improved to ~0.742-0.858 (mean ~0.847), which is better than the earlier ~0.999 regime but still not close to true numerical parity.
  - latest iteration (baseline-informed channels + longer parity fitting) remains in the same regime (~0.742-0.858, mean ~0.847), indicating a tuning plateau.
  - implemented first script-specific parity profile pass for `reservoir_simulation/2D/src` forward workflows:
    - workflow-aware loss/learning-rate scaling in `compat_runner.py`
    - deterministic baseline routing per workflow profile (prefers 2D-compatible `33331/UNRST.mat`)
    - wrappers now pass workflow identifiers (`Forward_problem_FNO`, `Forward_problem_PINO`, `Forward_problem_AFNOD`, `Forward_problem_AFNOP`)
  - extended script-specific profiling to all reservoir training wrappers (2D, 3D, CCUS, GenAI 2D/3D, Norne):
    - wrappers now pass both `workflow_name` and `scenario_name`
    - scenario-aware baseline candidate routing enabled in `compat_runner.py`
    - latest bridge-score snapshot by scenario:
      - 2D: mean ~0.858
      - 3D: mean ~0.742
      - CCUS: mean ~0.858
      - GenAI_2D: mean ~0.858
      - GenAI_3D: mean ~0.858
      - Norne: mean ~0.858
    - latest all-scenario bridge summary after GenAI_3D/Norne profile fix:
      - range ~0.742-0.858
      - mean ~0.847
  - bridge report now enforces quantitative gates per scenario/workflow and returns non-zero on regressions:
    - tightened scenario thresholds now enforced:
      - 2D: `<=0.87`
      - 3D: `<=0.77`
      - CCUS: `<=0.87`
      - GenAI_2D: `<=0.87`
      - GenAI_3D: `<=0.87`
      - Norne: `<=0.87`
    - current gate summary: `TOTAL=21 PASS=21 FAIL=0`
    - includes per-case gate line: `gate=score<=... status=PASS|FAIL`
    - includes scenario rollups under `GATE_SUMMARY`
  - parity matrix now runs the bridge gate as part of parity enforcement:
    - parity summary includes `BRIDGE_GATE=PASS|FAIL`
    - current result: `TOTAL=21 OK=21 FAIL=0`, `BRIDGE_GATE=PASS`
  - latest smoke matrix remains green after threshold tightening:
    - `TOTAL=29 OK=29 FAIL=0`

## Remaining Gaps

- Exact numerical parity with original reservoir pipelines is not established.
- Original data-loading and domain-specific preprocessing paths are not reinstated end-to-end in the rewritten wrappers.
- Original objective composition and model architecture parity (for each FNO/PINO/AFNO variant) is not yet restored.
- Baseline bridge confirms a remaining quantitative gap: predictions are directionally closer but still do not match historical baseline distributions.
- Artifact parity is not validated:
  - output tensor names/shapes against historical references
  - checkpoint compatibility with original downstream consumers
  - plotting/reporting outputs expected by legacy post-processing scripts

## Practical Decision Guide (Simple Terms)

- Important: the bridge `score` is a distance-like number, not a percent.
  - `0.0` would mean "identical distribution summary".
  - Smaller is better.
  - So `0.85` does **not** mean "85% correct".
- Current state is stable in the same score band (`~0.742-0.858`) across repeated runs, which suggests a tuning plateau under the current surrogate/compat setup.
- Practical implication:
  - If the goal is "examples run reliably with guarded regressions", current state is acceptable.
  - If the goal is "very close scientific reproduction of original workflows", further tuning of current knobs alone is unlikely to be enough; structural reconstruction work is needed.

## Bounded Improvement Strategy (No Endless Iterations)

Use a fixed-budget campaign with explicit stop rules:

1. Budget: at most 6 targeted experiments (not open-ended).
2. Scope per experiment: change one factor only (baseline routing, loss weights, or lr scale).
3. Keep-change rule: keep only if all conditions hold:
   - `compat_smoke_matrix.py`: all pass
   - `compat_parity_matrix.py`: all pass and `BRIDGE_GATE=PASS`
   - mean bridge score improves by at least `0.01` **or** worst-case scenario improves by at least `0.015`
4. Early stop rule: stop after 2 consecutive experiments with improvements below those thresholds.
5. Final decision after budget/early-stop:
   - If target improvement met: tighten gates and continue normal maintenance.
   - If not met: declare current compatibility mode "good enough for execution/regression protection" and defer true scientific parity to structural rebuild tasks.

## Recommended Call Right Now

- Recommendation: accept current state as "good enough" for runnable migrated examples and regression guarding, and avoid endless micro-tuning loops.
- Reason in plain terms: we already have strong run reliability and strict regression gates, while score movement has become marginal in repeated tuning cycles.
- Only continue aggressive parity work now if you explicitly need near-original scientific equivalence; in that case, proceed with data-path/objective/model reconstruction (not just more threshold and scale tweaks).

## Bounded Campaign Result (May 2026)

- Baseline before campaign:
  - mean `~0.847018`
  - best `~0.742085`
  - worst `~0.858074`
- Experiment 1 (single factor): increase PINO derivative weights `1.25 -> 1.35`
  - result: no measurable change (`mean ~0.847018`, worst unchanged)
  - decision: reject
- Experiment 2 (single factor): stronger lr dampening for GenAI_3D/Norne (`0.85 -> 0.75`)
  - result: tiny change only in 6th decimal (`mean ~0.847017`, worst unchanged)
  - decision: reject
- Early-stop trigger met:
  - two consecutive experiments below keep thresholds (`<0.01` mean gain and `<0.015` worst-case gain)
- Post-campaign state:
  - reverted to best-known stable profile
  - `compat_parity_matrix.py`: `TOTAL=21 OK=21 FAIL=0`, `BRIDGE_GATE=PASS`
  - `compat_smoke_matrix.py`: `TOTAL=29 OK=29 FAIL=0`
- Final verdict:
  - accept current compatibility mode as "good enough" for execution + regression protection
  - treat near-original scientific parity as a separate future track requiring structural rebuild work

## Next High-Impact Work

1. Replace compatibility target generation with direct historical-target supervision per workflow (script-specific pressure/saturation contracts).
2. Reconstruct per-script objective terms and schedules beyond generic weighted MSE + gradient losses.
3. Tighten gate thresholds scenario-by-scenario from current stabilization values as parity improves.
4. Build thresholded parity acceptance tests against designated baseline references, not nearest-baseline matching.
- Performance parity is not benchmarked:
  - throughput
  - memory footprint
  - convergence profile

## Next Practical Steps

1. Pick one family as baseline (recommended: `reservoir_simulation/2D/src`).
2. Reintroduce original data path and tensor contracts using existing local utilities.
3. Rebuild training losses to mirror each original script.
4. Add per-family parity checks (shape + metric thresholds).
5. Expand smoke matrix into parity matrix with expected-value tolerances.
