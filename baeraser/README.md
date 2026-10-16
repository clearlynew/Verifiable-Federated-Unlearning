# BAERASER-Inspired Trigger Recovery

This package implements a simplified, gradient-based trigger-recovery prototype for CIFAR-10 image-classification models. It searches candidate triggers, evaluates their attack success rates (ASR), and ranks the candidates. It does not perform unlearning.

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

The default trigger size is 6×6. Keep the training settings comparable between runs. The backdoored checkpoint includes trigger metadata for offline evaluation; do not provide that metadata to the recovery search.

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

## 3. Run the validation test

From the repository root:

```bash
python -m unittest -v baeraser.tests.test_recovery
```

The integration test trains or prepares controlled test models, runs recovery on both the backdoored model and the clean control, then compares the recovered result with the planted trigger's saved ground truth. It checks:

- recovered target label and trigger position;
- recovered-pattern similarity against the configured minimum;
- backdoored-model maximum ASR against the configured threshold;
- the ASR margin between the backdoored model and clean control; and
- the expected verification classification.

A passing test validates these checks for the controlled test case only. The full recovery search can take a substantial amount of time.

## Outputs

Recovery writes results to the directory specified by `--output`. Depending on the options used, this includes JSON summaries and candidate metadata, top candidate triggers in a PyTorch file, the recovery configuration, and clean-baseline/verdict results when `--clean-model` is provided.

## Limitations

This is a simplified BAERASER-inspired method, not a faithful implementation of the original BAERASER MSA/MESA and mutual-information mechanisms. It does not implement BAERASER's gradient-ascent unlearning stage.

A high ASR does not by itself prove that the original trigger was reconstructed. A clean model may also produce a non-zero maximum ASR during candidate search, so compare results against the clean baseline. Failure to recover a trigger does not prove that a model is backdoor-free.

The controlled experiment validates the recovery component, not successful federated unlearning or the complete verification pipeline. Test separately on actual post-unlearning models before making those claims.
