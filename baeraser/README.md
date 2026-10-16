# BAERASER-inspired Trigger Recovery

This package contains the team's gradient-based trigger-recovery prototype for investigating backdoor behavior in image-classification models.

## Scope and Limitations

This is a **simplified BAERASER-inspired method**, not a faithful implementation of the original BAERASER MSA/MESA and mutual-information mechanisms.

The implementation optimizes candidate triggers, evaluates their attack success rates (ASR), and ranks them using the configured evaluation criteria. High ASR indicates that a candidate can induce the target prediction; it does not, by itself, establish that the original trigger has been reconstructed.

This package currently focuses on trigger recovery and evaluation. Federated unlearning and verification are separate stages of the overall project.

## Files

- `recovery.py`: candidate optimization, trigger evaluation, ASR calculation, ranking, and optional clean-model baseline comparison.
- `__init__.py`: package initialization and public Python interface.
- `scripts/train_backdoor.py`: script for training a controlled backdoored model for experiments.
- `../model/model.py`: shared model definition.

The model definition is maintained in the shared `model/` package. Do not duplicate it inside this package.

## Dataset

The `--data-root` path must contain the `cifar-10-batches-py/` directory.

The recovery code uses the CIFAR-10 test split with `download=False`. The dataset must therefore already be available locally.

The training script uses CIFAR-10 to train a model with a controlled backdoor for experimental evaluation.

## Run Recovery

Run commands from the repository root.

```bash
python -m baeraser.recovery \
  --model /path/to/backdoored_model.pt \
  --data-root /path/to/cifar-10-root \
  --output /path/to/recovery_results
```

To evaluate a model alongside a clean-model baseline:

```bash
python -m baeraser.recovery \
  --model /path/to/model_under_test.pt \
  --clean-model /path/to/clean_model.pt \
  --data-root /path/to/cifar-10-root \
  --output /path/to/recovery_results
```

The model checkpoint must contain a compatible model state dictionary.

## Outputs

The recovery script saves:

- JSON summaries containing evaluation results.
- Candidate metadata, including target labels, trigger positions, and evaluation metrics.
- A PyTorch file containing the top candidate triggers in CIFAR-10 normalized input space.
- Additional clean-baseline results when `--clean-model` is provided.

## Evaluation

The main evaluation metrics include:

- **Attack Success Rate (ASR):** the fraction of eligible evaluation images classified as the candidate's target label after applying the trigger.
- **Clean Accuracy:** classification accuracy on unmodified evaluation images.
- **Trigger ranking:** comparison of candidates using the implemented evaluation criteria.

A clean-model comparison helps determine whether the recovered candidates behave differently on a model without the intended backdoor. Results should be interpreted cautiously: a candidate with high ASR is not necessarily visually or spatially identical to the original trigger.

## Current Status

This package is an experimental trigger-recovery component. Its results should not be described as verified unlearning or as a complete reproduction of the original BAERASER algorithm.
