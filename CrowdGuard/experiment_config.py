"""Configuration for the project CrowdGuard experiment.

The defaults in this file correspond to the project architecture.  The smoke
configuration is intentionally small so the same model/attack/detection code
can be exercised on a CPU before the full 20-client run.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ExperimentConfig:
    # Federated learning
    num_clients: int = 20
    samples_per_client: int = 2500
    local_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    rounds: int = 10
    dirichlet_alpha: float = 0.5
    partition: str = "iid"

    # Attack
    pmr: float = 0.05
    poison_rate: float = 0.10
    alpha: float = 0.70
    trigger_size: int = 6
    attack_start_round: int = 1
    target_label: int | None = None
    seed: int = 10

    # CrowdGuard / handoff
    # "frequency": client is in M if detected in >= final_m_tau * rounds rounds.
    # "union" was the provisional rule of the first run.
    final_m_rule: str = "frequency"
    final_m_tau: float = 0.5
    output_dir: str = "outputs"

    # Test-set split: kd_reference_size images (stratified) are kept aside as
    # unlabeled KD reference data for the later Wu stage; the rest is held out
    # for evaluation.  Keep test_split_seed fixed across runs so every run
    # (and every downstream stage) uses the identical split.
    kd_reference_size: int = 2500
    test_split_seed: int = 0

    # Commit-reveal self-test: corrupts one validator's reveal in every round
    # so the rejection path is exercised.  NEVER enable for real results.
    tamper_test: bool = False

    # Smoke-test controls
    smoke: bool = False

    def validate(self) -> None:
        if self.num_clients < 2:
            raise ValueError("num_clients must be >= 2")
        if not 0 <= self.pmr <= 1:
            raise ValueError("pmr must be in [0, 1]")
        if not 0 < self.poison_rate <= 1:
            raise ValueError("poison_rate must be in (0, 1]")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.trigger_size <= 0 or self.trigger_size > 32:
            raise ValueError("trigger_size must be between 1 and 32")
        if self.local_epochs <= 0 or self.batch_size <= 0 or self.rounds <= 0:
            raise ValueError("epochs, batch size, and rounds must be positive")
        if self.samples_per_client <= 0:
            raise ValueError("samples_per_client must be positive")
        if self.num_malicious_clients > self.num_clients:
            raise ValueError("number of malicious clients cannot exceed num_clients")
        if self.final_m_rule not in ("union", "frequency"):
            raise ValueError("final_m_rule must be 'union' or 'frequency'")
        if not 0 < self.final_m_tau <= 1:
            raise ValueError("final_m_tau must be in (0, 1]")
        if not 0 < self.kd_reference_size < 10000:
            raise ValueError("kd_reference_size must be between 1 and 9999")

    @property
    def num_malicious_clients(self) -> int:
        return max(1, int(self.num_clients * self.pmr)) if self.pmr > 0 else 0

    @property
    def num_benign_clients(self) -> int:
        return self.num_clients - self.num_malicious_clients

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["num_malicious_clients"] = self.num_malicious_clients
        result["num_benign_clients"] = self.num_benign_clients
        return result


def smoke_config() -> ExperimentConfig:
    """Small configuration for CPU validation of the full algorithmic path."""
    return ExperimentConfig(
        num_clients=4,
        samples_per_client=64,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.01,
        rounds=2,
        pmr=0.25,
        poison_rate=0.10,
        alpha=0.70,
        attack_start_round=1,
        seed=10,
        smoke=True,
    )
