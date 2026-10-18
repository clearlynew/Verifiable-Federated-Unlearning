# CrowdGuard: Federated Backdoor Detection and Handoff

## Overview

This repository implements and evaluates the **CrowdGuard detection stage** of a larger Verifiable Federated Unlearning project. The current work focuses on CIFAR-10 federated learning, backdoor detection, recording client-update history, and producing a validated handoff for downstream unlearning.

The current experiment is a **10-round detection-only study**. It is not the proposed future deployment workflow in which CrowdGuard runs periodically during a much longer training job.

### Current pipeline

```text
CIFAR-10 federated training
        |
        v
CrowdGuard-style HLBIM validation
        |
        v
Stacked-clustering aggregation of validator votes
        |
        v
Per-round detected-client sets M_t
        |
        v
Detection history + historical client-update capture
        |
        v
Final malicious-client set M
        |
        v
Validated handoff for downstream unlearning
```

The downstream stages are developed separately. This repository prepares the model, detection results, and historical information required by that workflow; it does not by itself establish that unlearning successfully removes a backdoor.

## Scope and implementation boundaries

The project adapts ideas from CrowdGuard and integrates them into a project-specific federated-learning and handoff pipeline. It should not be described as an exact reproduction of every detail of the original CrowdGuard paper.

This implementation includes:

- CIFAR-10 training with 20 logical clients.
- IID and Dirichlet-based data partition configurations.
- A lightweight CIFAR-10-compatible ResNet-18-style model.
- A scaling-based backdoor attack with a fixed-size pixel trigger.
- HLBIM-style hidden-layer analysis and stacked-clustering vote aggregation.
- Commit–reveal handling for validator votes.
- Detection history and a configurable rule for constructing the final malicious-client set.
- Historical client-update capture and handoff generation.
- Handoff integrity validation and post-hoc evaluation scripts/notebooks.

### Important deviation: detection-only aggregation

In the original CrowdGuard workflow, suspicious models are filtered before aggregation. In this project's **detection-only mode**, all participating client models remain in equal-weight FedAvg, including models flagged by CrowdGuard.

This is deliberate: the experiment preserves the potentially backdoored global model and the client history for the downstream unlearning stage. Therefore, do not describe the current aggregation step as removing malicious updates or as reproducing the original filtering behaviour.

## Experimental configuration

The standard configuration in `experiment_config.py` uses the following defaults. Individual handoffs record their run-specific configuration; use that metadata when reporting a particular experiment.

| Setting | Default |
|---|---:|
| Dataset | CIFAR-10 |
| Training set | 50,000 images |
| Held-out test set | 10,000 images before the evaluation split |
| Logical clients | 20 |
| Samples per client | 2,500 for the IID partition |
| Client participation | All clients each round |
| Federated rounds | 10 |
| Local epochs | 10 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Aggregation | Equal-weight FedAvg |
| Model | Lightweight ResNet-18-style |
| Trigger | 6 × 6 RGB pixel patch |
| Attack objective parameter (`alpha`) | 0.70 |
| Default PMR | 0.05 |
| Default seed | 10 |
| Final-set rule | Frequency |
| Default frequency threshold (`final_m_tau`) | 0.5 |
| KD-reference split | 2,500 test images, stratified |
| Remaining held-out evaluation split | 7,500 test images |

For non-IID runs, the partition is controlled by `partition="dirichlet"` and `dirichlet_alpha`. Do not infer a run's partition solely from a default configuration: prefer the metadata stored in its handoff, with the filename as a documented fallback for older handoffs that omit the partition field.

## Attack configuration

The experiment uses a project-specific constrain-and-scale backdoor attack. The attack configuration includes a clean classification objective, an anomaly/constrain component, trigger injection, and malicious-update scaling. The parameter `alpha` is part of the attack objective; it is distinct from the Dirichlet partition parameter `dirichlet_alpha`.

For 20 clients, the tested Poisoned Model Ratio (PMR) settings correspond to approximately:

| PMR | Malicious clients |
|---:|---:|
| 0.00 | 0 |
| 0.05 | 1 |
| 0.10 | 2 |
| 0.45 | 9 |

The implementation scales malicious updates approximately in proportion to `num_clients / num_malicious_clients`. Consequently, PMR changes both the number of malicious clients and the scaling applied to each malicious update. Performance differences across PMR settings should not be attributed to PMR alone without further controlled experiments.

The trigger is a 6 × 6 RGB patch. Its size is fixed for the experiment; the trigger position, RGB pattern, and target label are run-specific values recorded in the handoff as applicable. Evaluation ASR excludes test examples whose true label already equals the target label.

## Detection procedure

At each round, client models are analysed using hidden-layer representations. The HLBIM-style procedure computes distance-based features, applies PCA and statistical tests, and iteratively identifies suspicious models. Individual validator decisions are then combined using the project's stacked-clustering procedure.

The per-round result is a detected-client set, denoted `M_t`. Detection results are recorded for subsequent auditing and evaluation. Commit–reveal records support consistency checks on the reported votes; they should not be interpreted as proof that the underlying detector is accurate.

### Final malicious-client set

The current default is:

```python
final_m_rule = "frequency"
final_m_tau = 0.5
```

A client is included in the final set `M` when its detection frequency meets the configured threshold across the recorded rounds. With 10 rounds and `tau = 0.5`, this requires detection in at least 5 rounds. The repository also supports a `union` rule, which includes any client detected at least once.

Threshold sweeps are sensitivity analyses. If a threshold is chosen after inspecting these same experiments, report it as an empirical choice rather than independent validation. Keep round-level false-positive events separate from false-positive clients remaining in the final set `M`.

## Handoff and evaluation

The handoff preserves the final global model and the history needed by the downstream workflow. The exact schema is defined by the handoff utilities and validator in the repository.

The evaluation notebook operates on **saved handoffs**; it does not retrain the federated models. Its workflow is to:

1. Locate the intended `.pt` handoff files.
2. Run the repository's handoff-integrity validator.
3. Read configuration and detection metadata from each handoff.
4. Evaluate the initial and per-round global models on held-out clean accuracy and attack success rate (ASR).
5. Calculate per-round and final-set detection metrics and threshold-sensitivity tables.
6. Export CSV tables and figures for the report.

A successful handoff validation establishes internal consistency for the checks implemented by the validator. It does not independently prove detection effectiveness, unlearning effectiveness, or generalisation to unseen settings.

### Data split

The test set is divided into a stratified KD-reference subset and a separate held-out evaluation subset. The standard split uses 2,500 images for the KD reference data and 7,500 images for evaluation. Use the recorded split metadata when evaluating a handoff, and do not use the KD-reference subset as the held-out evaluation set.

### Recommended reporting outputs

For the main report, prioritize:

- A final-model table containing clean accuracy, ASR, and final-set detection metrics.
- Clean-accuracy and ASR trajectories by round, with readable legends and clear partition labels.
- A small number of representative round-level detection heatmaps.
- A compact comparison of final-set results at the prespecified threshold(s), with threshold-sensitivity results placed in the appendix.

Avoid presenting every generated plot in the main body. Clearly distinguish round-level detection events from the final set-level classification metrics.

## Repository files

Core implementation and evaluation files include:

- `cifar10_crowdguard.py` — federated training and CrowdGuard experiment entry point.
- `experiment_config.py` — experiment defaults and validation.
- `CrowdGuardClientValidation.py` and related detection modules — client-side validation logic.
- `commit_reveal.py` — commit–reveal vote handling.
- `handoff_utils.py` — handoff construction and supporting utilities.
- `validate_handoff.py` — handoff integrity checks.
- `evaluate_handoff.py` — evaluation of saved handoffs.
- `ExperimentMatrix.md` — experiment configurations and their status.
- The handoff-evaluation notebook — batch validation, evaluation, tables, and report figures for saved `.pt` files.

Use the filenames present in the checked-out branch; notebook filenames may differ between versions.

## Running and evaluating

Run commands from the repository root and use the Python environment configured for the project. Before a full experiment, use the project's smoke configuration or tests to check the pipeline on a small workload.

A typical training invocation is:

```bash
python cifar10_crowdguard.py --help
```

Use the arguments shown by the checked-out version of the script to launch a run; do not assume that every configuration field is exposed as a command-line argument. Record the Git commit, run configuration, and generated handoff for reproducibility.

To validate a saved handoff:

```bash
python validate_handoff.py path/to/handoff.pt
```

To evaluate a saved handoff, use the options supported by the checked-out version of `evaluate_handoff.py`. The evaluation notebook automates this across multiple handoffs and exports consolidated results.

## Runtime and resource considerations

A full experiment can be computationally expensive because every round involves local training for 20 clients as well as detection/validation work. Post-hoc handoff evaluation avoids retraining, but it still performs inference over the saved global models and can take time across many handoffs. Run evaluation on the available CPU/GPU and avoid rerunning training solely to regenerate tables or figures when valid handoffs already exist.

## Limitations

- The current main protocol uses 10 rounds; it does not evaluate a long-running FL deployment with periodic detection and remediation.
- Detection-only aggregation intentionally retains flagged updates.
- Results are limited to the completed configurations listed in `ExperimentMatrix.md`.
- Some strongly non-IID (`Dirichlet α = 0.1`) runs were not completed due to process termination; do not report results for those runs unless they are successfully rerun and validated.
- Multiple seeds are available only for selected IID configurations. These are limited robustness checks, not broad estimates of variance.
- PMR settings change both the number of malicious clients and malicious-update scaling; shard assignments may also differ between runs.
- A high detection score does not establish successful unlearning. Post-unlearning clean accuracy and ASR must be evaluated by the downstream stage.

## Reference

Rieger, P., Krauß, T., Miettinen, M., Dmitrienko, A., & Sadeghi, A. R. “CrowdGuard: Federated Backdoor Detection in Federated Learning.” NDSS 2024.
