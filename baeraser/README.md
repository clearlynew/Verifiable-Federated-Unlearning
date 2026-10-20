# BAERASER-Inspired Trigger Recovery

This package implements a simplified, gradient-based trigger-recovery prototype for CIFAR-10 image-classification models. It searches candidate triggers, evaluates their attack success rates (ASR), and ranks the candidates. It does not perform unlearning.

## Folder layout

```text
Verifiable-Federated-Unlearning/        <- repository root (run everything from here)
├── model/
│   ├── __init__.py                      shared team model package (unchanged)
│   └── model.py                         LightweightResNet18 (unchanged)
└── baeraser/
    ├── __init__.py                      lazy exports of the public API
    ├── requirements.txt
    ├── README.md                        this file
    ├── .gitignore
    ├── recovery.py                      the recovery + verdict module and CLI
    ├── scripts/
    │   └── train_backdoor.py            makes controlled backdoored / clean test models
    └── tests/
        ├── __init__.py
        └── test_recovery.py             integration test (skipped unless env vars are set)
```

## Requirements

- Run commands from the repository root.
- The shared model definition must be available at `model/model.py`.
- CIFAR-10 must already be downloaded. `DATA_ROOT` must contain `cifar-10-batches-py/`.
- Install dependencies from `baeraser/requirements.txt`.

## 1. Train the test models

Use `baeraser/scripts/train_backdoor.py` to create controlled models for validating recovery. These are centralized test models, not federated or post-unlearning models.

Set `DATA_ROOT` to the directory containing `cifar-10-batches-py/`.

**Backdoored model:**

```bash
python baeraser/scripts/train_backdoor.py \
  --data "$DATA_ROOT" \
  --out /path/to/backdoored_model.pt \
  --epochs 15 \
  --poison-rate 0.10
```

**Clean control model:**

```bash
python baeraser/scripts/train_backdoor.py \
  --data "$DATA_ROOT" \
  --out /path/to/clean_model.pt \
  --epochs 15 \
  --poison-rate 0.0
```

The default trigger size is 6×6. Use `--seed N` to plant a different trigger position and target label (the seed also fixes the random trigger pattern). Keep the training settings comparable between runs. The backdoored checkpoint includes trigger metadata for offline evaluation; do not provide that metadata to the recovery search.

## 2. Run recovery

Run from the repository root:

```bash
python -m baeraser.recovery \
  --model /path/to/backdoored_model.pt \
  --data-root "$DATA_ROOT" \
  --output /path/to/recovery_results
```

To compare the backdoored model with the clean control:

```bash
python -m baeraser.recovery \
  --model /path/to/backdoored_model.pt \
  --clean-model /path/to/clean_model.pt \
  --data-root "$DATA_ROOT" \
  --output /path/to/recovery_results
```

The checkpoints must be compatible with the shared `LightweightResNet18` architecture.

The verifier's data is split into two **disjoint** parts: the first `--optimization-images` (default 32) images are used only to optimize candidate triggers, and all remaining images are used only to score them (ASR and clean accuracy). `--verification-images` must therefore be larger than `--optimization-images`. With the default 1,000 verification images, 32 are used for optimization and 968 for scoring (fewer per candidate, because images of the candidate's own target class are excluded from ASR).

**Debug only.** `--debug-locations U,V [U,V ...]` and `--debug-labels L [L ...]` restrict the search so a smoke run finishes in minutes. A restricted run prints a warning and its verdict carries `"valid_protocol_result": false`. It is never a protocol result.

## 3. Run the validation test

From the repository root:

```bash
python -m unittest -v baeraser.tests.test_recovery
```

The test is skipped unless these environment variables are set:

| Variable | Meaning |
|---|---|
| `BAERASER_TEST_MODEL` | backdoored checkpoint from `train_backdoor.py` (must contain `trigger_metadata`) |
| `BAERASER_TEST_CLEAN_MODEL` | clean control checkpoint |
| `BAERASER_TEST_DATA_ROOT` | directory containing `cifar-10-batches-py/` |
| `BAERASER_TEST_OUTPUT` | output directory (default `baeraser_validation_results`) |
| `BAERASER_TEST_EXTRA_ARGS` | optional extra `recovery.py` arguments, for smoke runs only (e.g. `--debug-locations 1,13 5,5`) |
| `BAERASER_MIN_PATTERN_SIMILARITY` | optional; if set, adds a pattern-similarity assertion (see note below) |
| `BAERASER_TEST_*` (others) | `TOP_N`, `OPTIMIZATION_STEPS`, `VERIFICATION_IMAGES`, `OPTIMIZATION_IMAGES`, `CHUNK_SIZE`, `NUM_WORKERS`, `TIMEOUT_SECONDS` |

The integration test runs recovery on the backdoored model and the clean control, then compares the recovered result with the planted trigger's saved ground truth. It checks:

- a candidate with the true target label **and** true position is among the saved top candidates;
- backdoored-model maximum ASR against the configured threshold;
- the ASR margin between the backdoored model and clean control; and
- the expected verification classification (`RESIDUAL_BACKDOOR_EVIDENCE`).

It also **reports** (in `recovery_validation_metrics.json`) the top-1 label and its position error, the pattern similarity, and the chance level of that similarity.

> **Pattern similarity is reported, not asserted by default.** On this metric two *unrelated* random patterns score about 0.67 (never below about 0.60), so a fixed gate such as 0.60 would not discriminate. Recovery runs only 15 Adam steps from a random start, so a recovered trigger is expected to behave like the real one (similar ASR) without matching its pixels. Compare the reported similarity with `pattern_similarity_chance_level`; a value near chance means no pixel-level match.

A passing test validates these checks for the controlled test case only. The full recovery search (729 positions x 10 labels = 7,290 candidates) can take a substantial amount of time; use a GPU.

## 4. Using recovery from other code (per-client verification)

```python
from baeraser.recovery import (
    RecoveryConfig, verify_client, classify_verification,
)

config = RecoveryConfig()                  # protocol defaults: tau=0.80, delta=0.10
verdict = verify_client(
    model_u,            # post-unlearning model (LightweightResNet18, eval-able)
    clean_model,        # same architecture, never backdoored
    client_loader,      # this client's local clean data (shuffle=False)
    config,
)
verdict["status"]       # RESIDUAL_BACKDOOR_EVIDENCE | NO_RECOVERABLE_TRIGGER_UNDER_PROTOCOL
                        # | INCONCLUSIVE_OR_BASELINE-COMPATIBLE
                        # | INCONCLUSIVE_INSUFFICIENT_VERIFIER_DATA
verdict["vote_bit"]     # 1 residual evidence, 0 none / baseline-compatible, None = inconclusive
```

`vote_bit` is only a default mapping. The vote encoding and tally rule belong to the commit-reveal design and should be agreed with whoever owns it. If the clean-baseline summary for the same data and config is already computed, pass `clean_summary=` instead of `clean_model`.

Offline evaluation against the hidden trigger (never pass this to the verifier):

```python
from baeraser.recovery import GroundTruthTrigger, evaluate_against_ground_truth
gt = GroundTruthTrigger(u=py, v=px, target_label=t, pattern=normalized_pattern)  # [3,6,6]
evaluate_against_ground_truth(verdict["recovery_results"], gt, model_u, eval_loader, device)
```

## Outputs

Recovery writes results to the directory specified by `--output`:

| File | Contents |
|---|---|
| `post_unlearning_results.json` | every candidate (label, position, ASR, clean accuracy), sorted by ASR; a candidate's rank is its index + 1 |
| `post_unlearning_top_triggers.pt` | the 50 highest-ASR triggers and their metadata |
| `post_unlearning_summary.json` | maximum ASR, best candidate, number of candidates, number above tau |
| `clean_baseline_results.json`, `clean_baseline_top_triggers.pt`, `clean_baseline_summary.json` | the same for the clean control (only with `--clean-model`) |
| `verification_verdict.json` | status, both maximum ASRs, `valid_protocol_result` (only with `--clean-model`) |
| `recovery_config.json` | the configuration used |

On a Kaggle Tesla T4, one full recovery (7,290 candidates) took about 9.5 minutes per model.

## Limitations

This is a simplified BAERASER-inspired method, not a faithful implementation of the original BAERASER MSA/MESA and mutual-information mechanisms. It does not implement BAERASER's gradient-ascent unlearning stage.

A high ASR does not by itself prove that the original trigger was reconstructed. A clean model may also produce a non-zero maximum ASR during candidate search, so compare results against the clean baseline. Failure to recover a trigger does not prove that a model is backdoor-free.

The controlled experiment validates the recovery component, not successful federated unlearning or the complete verification pipeline. Test separately on actual post-unlearning models before making those claims.
