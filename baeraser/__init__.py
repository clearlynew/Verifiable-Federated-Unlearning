"""BAERASER-inspired trigger-recovery package.

Names are exported lazily so that ``python -m baeraser.recovery`` does not
import ``baeraser.recovery`` twice (which triggers a RuntimeWarning).
"""

__all__ = [
    "RecoveryConfig",
    "GroundTruthTrigger",
    "recover_triggers",
    "summarize_recovery",
    "classify_verification",
    "verify_client",
    "evaluate_against_ground_truth",
]


def __getattr__(name):
    if name in __all__:
        from . import recovery

        return getattr(recovery, name)
    raise AttributeError(f"module 'baeraser' has no attribute {name!r}")
