"""Experimental integration test for BAERASER-inspired trigger recovery.

This is a validation test, not merely a CLI/output smoke test. It runs recovery
on a controlled backdoored checkpoint and a clean checkpoint, then checks:
  1. the ground-truth target and location are represented among top candidates;
  2. the recovered pattern meets a configurable offline similarity threshold;
  3. the backdoored model produces residual-backdoor evidence relative to the
     clean-model recovery baseline.

Run from the repository root with the environment variables documented in
baeraser/report/baeraser_phase1_review2.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

# Test paths are explicit so this can run in Colab or another configured
# environment without hard-coding a particular notebook directory.
MODEL_PATH = os.environ.get("BAERASER_TEST_MODEL")
CLEAN_MODEL_PATH = os.environ.get("BAERASER_TEST_CLEAN_MODEL")
DATA_ROOT = os.environ.get("BAERASER_TEST_DATA_ROOT")
OUTPUT_DIR = Path(os.environ.get("BAERASER_TEST_OUTPUT", "baeraser_validation_results"))

# This threshold is an experimental acceptance criterion, not a claim about
# the original BAERASER paper. Override only when the review protocol justifies it.
MIN_PATTERN_SIMILARITY = float(
    os.environ.get("BAERASER_MIN_PATTERN_SIMILARITY", "0.60")
)
TOP_N = int(os.environ.get("BAERASER_TEST_TOP_N", "50"))
OPTIMIZATION_STEPS = int(os.environ.get("BAERASER_TEST_OPTIMIZATION_STEPS", "15"))
VERIFICATION_IMAGES = int(os.environ.get("BAERASER_TEST_VERIFICATION_IMAGES", "1000"))
OPTIMIZATION_IMAGES = int(os.environ.get("BAERASER_TEST_OPTIMIZATION_IMAGES", "32"))
CANDIDATE_CHUNK_SIZE = int(os.environ.get("BAERASER_TEST_CHUNK_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("BAERASER_TEST_NUM_WORKERS", "2"))
TIMEOUT_SECONDS = int(os.environ.get("BAERASER_TEST_TIMEOUT_SECONDS", "7200"))


def _load_checkpoint(path: str):
    """Load the experiment checkpoint including offline ground-truth metadata."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


class RecoveryValidationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = []
        if not MODEL_PATH:
            missing.append("BAERASER_TEST_MODEL")
        if not CLEAN_MODEL_PATH:
            missing.append("BAERASER_TEST_CLEAN_MODEL")
        if not DATA_ROOT:
            missing.append("BAERASER_TEST_DATA_ROOT")
        if missing:
            raise unittest.SkipTest(
                "Set required environment variables to run the experimental "
                "validation: " + ", ".join(missing)
            )

        for label, path in (
            ("backdoored checkpoint", MODEL_PATH),
            ("clean checkpoint", CLEAN_MODEL_PATH),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        batch_dir = Path(DATA_ROOT) / "cifar-10-batches-py"
        if not batch_dir.is_dir():
            raise FileNotFoundError(
                f"CIFAR-10 directory not found: {batch_dir}. "
                "BAERASER_TEST_DATA_ROOT must be its parent directory."
            )

        checkpoint = _load_checkpoint(MODEL_PATH)
        if not isinstance(checkpoint, dict) or "trigger_metadata" not in checkpoint:
            raise AssertionError(
                "Backdoored checkpoint must include trigger_metadata from "
                "scripts/train_backdoor.py; ground-truth recovery cannot be assessed."
            )
        cls.ground_truth = checkpoint["trigger_metadata"]

        required = {"position", "target_label", "pattern_raw_0_1", "size"}
        missing_metadata = required.difference(cls.ground_truth)
        if missing_metadata:
            raise AssertionError(
                "Checkpoint trigger_metadata is missing: "
                + ", ".join(sorted(missing_metadata))
            )

        cls.output_dir = OUTPUT_DIR.resolve()
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_recovery_quality_and_clean_model_negative_control(self):
        """Run full recovery and validate recovered trigger plus clean baseline."""
        command = [
            sys.executable,
            "-W",
            "ignore::RuntimeWarning",
            "-u",
            str(Path(__file__).resolve().parents[1] / "recovery.py"),
            "--model", str(Path(MODEL_PATH).resolve()),
            "--clean-model", str(Path(CLEAN_MODEL_PATH).resolve()),
            "--data-root", str(Path(DATA_ROOT).resolve()),
            "--output", str(self.output_dir),
            "--optimization-steps", str(OPTIMIZATION_STEPS),
            "--optimization-images", str(OPTIMIZATION_IMAGES),
            "--candidate-chunk-size", str(CANDIDATE_CHUNK_SIZE),
            "--verification-images", str(VERIFICATION_IMAGES),
            "--num-workers", str(NUM_WORKERS),
        ]

        log_path = self.output_dir / "integration_test_recovery.log"
        # The recovery CLI already prints per-label and per-location progress.
        # Run Python unbuffered and forward each output line immediately so
        # Colab displays progress while the long integration test is running.
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        # The recovery CLI prints the actual selected device once, so avoid
        # duplicating device information in the test harness.
        startup_lines = [
            "Starting recovery validation...",
            "Running recovery against the test model and clean control.",
            "Progress updates show candidate locations and elapsed time.",
            "",
        ]
        print("\n".join(startup_lines), flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("\n".join(startup_lines) + "\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=child_env,
                text=True,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output_lines = []
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()
                    output_lines.append(line)
                return_code = process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.fail(
                    f"Recovery CLI exceeded timeout of {TIMEOUT_SECONDS} seconds. "
                    f"See live log: {log_path}"
                )

        output_text = "".join(output_lines)
        self.assertEqual(
            return_code,
            0,
            f"Recovery CLI failed with exit code {return_code}. "
            f"See {log_path}\n\nLast output:\n{output_text[-6000:]}",
        )

        # Validate the actual recovery results, not only the existence of files.
        results_path = self.output_dir / "post_unlearning_results.json"
        triggers_path = self.output_dir / "post_unlearning_top_triggers.pt"
        summary_path = self.output_dir / "post_unlearning_summary.json"
        clean_summary_path = self.output_dir / "clean_baseline_summary.json"
        verdict_path = self.output_dir / "verification_verdict.json"

        for path in (
            results_path, triggers_path, summary_path, clean_summary_path, verdict_path
        ):
            self.assertTrue(path.is_file(), f"Expected output was not created: {path}")

        results = json.loads(results_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        clean_summary = json.loads(clean_summary_path.read_text(encoding="utf-8"))
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        top_payload = _load_checkpoint(str(triggers_path))

        self.assertTrue(results, "Recovery produced no candidate results.")
        self.assertGreaterEqual(
            len(top_payload.get("metadata", [])), 1,
            "No candidate trigger metadata was saved.",
        )
        self.assertIn("max_asr", summary)
        self.assertIn("max_asr", clean_summary)

        # Identify the best saved candidate that matches both the known target
        # and the known top-left position. Ground truth is used ONLY for this
        # offline test, never by recovery itself.
        gt_target = int(self.ground_truth["target_label"])
        gt_position = tuple(int(v) for v in self.ground_truth["position"])
        gt_size = int(self.ground_truth["size"])
        self.assertEqual(gt_size, 6, "Current recovery protocol assumes a 6x6 trigger.")

        matching = [
            (rank, meta)
            for rank, meta in enumerate(top_payload["metadata"], start=1)
            if int(meta["target_label"]) == gt_target
            and (int(meta["u"]), int(meta["v"])) == gt_position
        ]
        self.assertTrue(
            matching,
            "No top-ranked candidate matched both the ground-truth target label "
            f"{gt_target} and position {gt_position}. See {results_path}.",
        )

        rank, matched_meta = matching[0]
        recovered = top_payload["triggers"][rank - 1].detach().cpu().float()

        # The checkpoint stores the true trigger in raw RGB [0,1]. The recovery
        # trigger is stored in CIFAR-10 normalized input space; compare in the
        # same normalized space, using the recovery module's normalization bounds.
        raw = torch.as_tensor(
            self.ground_truth["pattern_raw_0_1"], dtype=torch.float32
        )
        mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.2470, 0.2435, 0.2616), dtype=torch.float32).view(3, 1, 1)
        ground_truth_normalized = (raw - mean) / std

        # Import after the CLI run so this assertion uses the package in the repo.
        from baeraser.recovery import normalized_input_bounds, pattern_similarity

        lo, hi = normalized_input_bounds(device=torch.device("cpu"))
        similarity = pattern_similarity(
            recovered, ground_truth_normalized, lo.cpu(), hi.cpu()
        )

        metrics = {
            "matched_rank": rank,
            "ground_truth_target_label": gt_target,
            "recovered_target_label": int(matched_meta["target_label"]),
            "ground_truth_position": list(gt_position),
            "recovered_position": [int(matched_meta["u"]), int(matched_meta["v"])],
            "position_error_pixels": 0.0,
            "pattern_similarity": similarity,
            "minimum_pattern_similarity": MIN_PATTERN_SIMILARITY,
            "backdoored_model_max_asr": float(summary["max_asr"]),
            "clean_model_max_asr": float(clean_summary["max_asr"]),
            "asr_margin": float(summary["max_asr"] - clean_summary["max_asr"]),
            "verification_verdict": verdict.get("status"),
            "passes_tau": verdict.get("passes_tau"),
            "beats_clean_baseline": verdict.get("beats_clean_baseline"),
        }
        metrics_path = self.output_dir / "recovery_validation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\nBAERASER experimental validation metrics:")
        print(json.dumps(metrics, indent=2))

        self.assertGreaterEqual(
            similarity,
            MIN_PATTERN_SIMILARITY,
            f"Exact-position/target candidate pattern similarity {similarity:.3f} "
            f"is below threshold {MIN_PATTERN_SIMILARITY:.3f}. "
            f"See {metrics_path}.",
        )

        # This validates discrimination against the clean model using the
        # implementation's preconfigured tau-ASR and baseline-margin decision.
        self.assertEqual(
            verdict.get("status"),
            "RESIDUAL_BACKDOOR_EVIDENCE",
            "Backdoored model did not satisfy the configured ASR and clean-baseline "
            f"margin criteria. Verdict: {verdict}. See {metrics_path}.",
        )
        self.assertTrue(verdict.get("passes_tau"), "Backdoored model did not pass tau-ASR.")
        self.assertTrue(
            verdict.get("beats_clean_baseline"),
            "Backdoored model did not exceed the clean-model baseline by delta_baseline.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=0)
