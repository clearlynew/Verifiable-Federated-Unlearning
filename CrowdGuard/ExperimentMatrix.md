# Experiment Matrix

This matrix separates the configurations represented by the current saved handoffs from configurations that were planned but not completed. The handoff-evaluation notebook is configured to process the ten saved handoffs listed below; use the metadata inside each `.pt` file as the source of truth for the actual run configuration.

## Completed handoffs

| Data partition | Attacker | PMR | Seed | Status / purpose |
|---|---|---:|---:|---|
| IID | No | 0.00 | 10 | Clean baseline |
| IID | Yes | 0.05 | 10 | Low-PMR attack |
| IID | Yes | 0.10 | 10 | Main attack condition |
| IID | Yes | 0.45 | 10 | High-PMR stress condition |
| IID | No | 0.00 | 42 | Clean-seed robustness check |
| IID | Yes | 0.05 | 42 | Low-PMR seed robustness check |
| IID | Yes | 0.10 | 42 | Main-attack seed robustness check |
| Dirichlet α = 0.5 | No | 0.00 | 10 | Moderate heterogeneity baseline |
| Dirichlet α = 0.5 | Yes | 0.10 | 10 | Moderate heterogeneity + main attack |
| Dirichlet α = 0.5 | Yes | 0.45 | 10 | Moderate heterogeneity + stress condition |

## Planned but not completed

| Data partition | Attacker | PMR | Seed | Status / note |
|---|---|---:|---:|---|
| Dirichlet α = 0.1 | No | 0.00 | 10 | Not completed; process terminated during execution |
| Dirichlet α = 0.1 | Yes | 0.10 | 10 | Not completed; process terminated during execution |
| Dirichlet α = 0.1 | Yes | 0.45 | 10 | Not completed; process terminated during execution |

The observed process exit code was `-9`; this is consistent with an external kill and may be caused by memory pressure, but the exact cause was not confirmed. Do not report detection or accuracy results for these configurations.

## Common configuration

Unless the individual handoff metadata says otherwise, the experiment protocol uses:

| Setting | Value |
|---|---:|
| Dataset | CIFAR-10 |
| Logical clients | 20 |
| Federated rounds | 10 |
| Client participation | All clients every round |
| Aggregation | Equal-weight FedAvg |
| Local epochs | 10 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Model | Lightweight ResNet-18-style |
| Trigger | 6 × 6 RGB pixel patch |
| Attack objective parameter (`alpha`) | 0.70 |
| Poison rate | 0.10 |
| Final malicious-set rule | Frequency |
| Configured frequency threshold (`final_m_tau`) | 0.5 |
| KD-reference subset | 2,500 stratified test images |
| Held-out evaluation subset | 7,500 test images |

For IID runs, 20 clients receive 2,500 training samples each. Dirichlet partitioning changes the per-client label distribution; do not assume that each non-IID client has exactly 2,500 examples unless the recorded configuration confirms it.

## Interpretation notes

- **PMR** is the fraction of clients designated malicious. With 20 clients, PMR 0.05, 0.10, and 0.45 correspond to 1, 2, and 9 malicious clients, respectively.
- The implementation scales malicious updates approximately as `num_clients / num_malicious_clients`. Thus PMR changes both the number of malicious clients and per-client attack scaling.
- All clients participate in every round. CrowdGuard detections are recorded, but flagged client models are still included in FedAvg in the current detection-only experiment.
- `M_t` denotes the set detected in round `t`; `M` denotes the final malicious-client set selected from the recorded detection history.
- The default final-set rule is frequency-based with `tau = 0.5`. A threshold sweep is a sensitivity analysis; selecting a threshold after inspecting the same experiments is not independent validation.
- Round-level false-positive events and false-positive clients in the final set `M` are different metrics and must be reported separately.
- The attack starts at round 1 in the saved configurations. Round 0 is pre-attack; a malicious ground-truth client flagged in round 0 is not a round-0 attack detection, and a failure to flag it in round 0 should not be counted as an attack-active false negative.
- The test set is split into a KD-reference subset and a separate held-out evaluation subset. ASR excludes examples whose true class is already the target class.
- The current experiments evaluate a 10-round detection-only protocol. Periodic detection in a longer FL run and automatic unlearning/remediation are future-work directions, not results established by this matrix.

## Evaluation artifacts

The handoff-evaluation notebook validates and evaluates saved `.pt` handoffs without retraining. Its outputs include per-round model metrics, detection metrics, threshold-sensitivity tables, final-set summaries, and figures. Review labels and partition metadata before using any output in the report; older handoffs may omit a partition field, in which case the notebook infers the partition from the filename and should preserve that inference transparently.
