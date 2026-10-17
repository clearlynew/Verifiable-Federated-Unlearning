"""End-to-end validation for BAERASER-inspired trigger recovery.

Runs recovery on a controlled backdoored checkpoint and a clean checkpoint.
ASR-based verification is the pass/fail criterion. Ground-truth trigger
position/label and pattern similarity are offline diagnostics only.

Required environment variables:
  BAERASER_TEST_MODEL, BAERASER_TEST_CLEAN_MODEL, BAERASER_TEST_DATA_ROOT
Optional tuning variables are listed below.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

MODEL_PATH = os.environ.get("BAERASER_TEST_MODEL")
CLEAN_MODEL_PATH = os.environ.get("BAERASER_TEST_CLEAN_MODEL")
DATA_ROOT = os.environ.get("BAERASER_TEST_DATA_ROOT")
OUTPUT_DIR = Path(os.environ.get("BAERASER_TEST_OUTPUT", "baeraser_validation_results"))
MIN_PATTERN_SIMILARITY = float(os.environ.get("BAERASER_MIN_PATTERN_SIMILARITY", "0.60"))
OPTIMIZATION_STEPS = int(os.environ.get("BAERASER_TEST_OPTIMIZATION_STEPS", "15"))
VERIFICATION_IMAGES = int(os.environ.get("BAERASER_TEST_VERIFICATION_IMAGES", "1000"))
OPTIMIZATION_IMAGES = int(os.environ.get("BAERASER_TEST_OPTIMIZATION_IMAGES", "32"))
CANDIDATE_CHUNK_SIZE = int(os.environ.get("BAERASER_TEST_CHUNK_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("BAERASER_TEST_NUM_WORKERS", "2"))
TIMEOUT_SECONDS = int(os.environ.get("BAERASER_TEST_TIMEOUT_SECONDS", "7200"))


def _load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RecoveryValidationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name, value in (
            ("BAERASER_TEST_MODEL", MODEL_PATH),
            ("BAERASER_TEST_CLEAN_MODEL", CLEAN_MODEL_PATH),
            ("BAERASER_TEST_DATA_ROOT", DATA_ROOT),
        ) if not value]
        if missing:
            raise unittest.SkipTest("Set required environment variables: " + ", ".join(missing))
        for label, path in (("backdoored checkpoint", MODEL_PATH), ("clean checkpoint", CLEAN_MODEL_PATH)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        if not (Path(DATA_ROOT) / "cifar-10-batches-py").is_dir():
            raise FileNotFoundError("BAERASER_TEST_DATA_ROOT must contain cifar-10-batches-py/")
        checkpoint = _load_checkpoint(MODEL_PATH)
        if not isinstance(checkpoint, dict) or "trigger_metadata" not in checkpoint:
            raise AssertionError("Backdoored checkpoint must include trigger_metadata.")
        cls.ground_truth = checkpoint["trigger_metadata"]
        required = {"position", "target_label", "pattern_raw_0_1", "size"}
        missing_metadata = required.difference(cls.ground_truth)
        if missing_metadata:
            raise AssertionError("Checkpoint trigger_metadata missing: " + ", ".join(sorted(missing_metadata)))
        cls.output_dir = OUTPUT_DIR.resolve()
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def test_recovery_quality_and_clean_model_negative_control(self):
        repo_root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable, "-W", "ignore::RuntimeWarning", "-u",
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
        print("Starting recovery validation (backdoored model and clean control)...", flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command, cwd=repo_root, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()
                return_code = process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.fail(f"Recovery exceeded {TIMEOUT_SECONDS}s. See {log_path}")
        self.assertEqual(return_code, 0, f"Recovery failed. See {log_path}")

        out = self.output_dir
        required_files = [
            "post_unlearning_results.json", "post_unlearning_all_triggers.pt",
            "post_unlearning_summary.json", "clean_baseline_results.json",
            "clean_baseline_all_triggers.pt", "clean_baseline_summary.json",
            "verification_verdict.json",
        ]
        for filename in required_files:
            self.assertTrue((out / filename).is_file(), f"Missing output: {filename}")

        results = json.loads((out / "post_unlearning_results.json").read_text())
        summary = json.loads((out / "post_unlearning_summary.json").read_text())
        clean_results = json.loads((out / "clean_baseline_results.json").read_text())
        clean_summary = json.loads((out / "clean_baseline_summary.json").read_text())
        verdict = json.loads((out / "verification_verdict.json").read_text())
        payload = _load_checkpoint(str(out / "post_unlearning_all_triggers.pt"))
        self.assertEqual(len(results), 7290, "Expected 729 locations x 10 labels.")
        self.assertEqual(len(clean_results), 7290, "Expected 729 clean-control candidates per label set.")
        self.assertEqual(len(payload["metadata"]), len(results))
        self.assertEqual(int(summary["num_candidates"]), len(results))

        gt_target = int(self.ground_truth["target_label"])
        gt_position = tuple(int(v) for v in self.ground_truth["position"])
        self.assertEqual(int(self.ground_truth["size"]), 6, "Protocol expects a 6x6 trigger.")
        matches = [
            (i, meta) for i, meta in enumerate(payload["metadata"])
            if int(meta["target_label"]) == gt_target
            and (int(meta["u"]), int(meta["v"])) == gt_position
        ]
        self.assertEqual(len(matches), 1, "Expected exactly one candidate at the planted target/position.")
        idx, matched_meta = matches[0]
        rank = int(matched_meta.get("rank", idx + 1))
        recovered = payload["triggers"][idx].detach().cpu().float()

        raw = torch.as_tensor(self.ground_truth["pattern_raw_0_1"], dtype=torch.float32)
        mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
        std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
        ground_truth_normalized = (raw - mean) / std
        from baeraser.recovery import normalized_input_bounds, pattern_similarity
        lo, hi = normalized_input_bounds(device=torch.device("cpu"))
        similarity = pattern_similarity(recovered, ground_truth_normalized, lo, hi)

        best = results[0]
        clean_best = clean_results[0]
        metrics = {
            "candidate_count_per_model": len(results),
            "optimization_steps": OPTIMIZATION_STEPS,
            "optimization_images": OPTIMIZATION_IMAGES,
            "verification_images": VERIFICATION_IMAGES,
            "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
            "backdoored_model": {
                "max_asr": float(summary["max_asr"]),
                "best_target_label": int(best["target_label"]),
                "best_position": [int(best["u"]), int(best["v"])],
                "best_clean_accuracy": float(best["clean_accuracy"]),
                "candidates_at_or_above_tau": int(summary["num_above_tau"]),
            },
            "clean_control": {
                "max_asr": float(clean_summary["max_asr"]),
                "best_target_label": int(clean_best["target_label"]),
                "best_position": [int(clean_best["u"]), int(clean_best["v"])],
                "best_clean_accuracy": float(clean_best["clean_accuracy"]),
                "candidates_at_or_above_tau": int(clean_summary["num_above_tau"]),
            },
            "asr_margin": float(summary["max_asr"] - clean_summary["max_asr"]),
            "verification_verdict": verdict.get("status"),
            "passes_tau": verdict.get("passes_tau"),
            "beats_clean_baseline": verdict.get("beats_clean_baseline"),
            "ground_truth_diagnostics": {
                "target_label": gt_target,
                "position": list(gt_position),
                "candidate_rank_by_asr": rank,
                "is_top_ranked_candidate": rank == 1,
                "candidate_asr": float(matched_meta["asr"]),
                "pattern_similarity_1_minus_normalized_mae": similarity,
                "similarity_reference_threshold": MIN_PATTERN_SIMILARITY,
                "similarity_meets_reference_threshold": similarity >= MIN_PATTERN_SIMILARITY,
                "threshold_is_diagnostic_only": True,
            },
        }
        metrics_path = out / "recovery_validation_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\nBAERASER validation metrics (similarity is diagnostic only):")
        print(json.dumps(metrics, indent=2), flush=True)

        # Verification pass/fail uses ASR and clean-baseline criteria only.
        self.assertEqual(verdict.get("status"), "RESIDUAL_BACKDOOR_EVIDENCE", f"Verdict: {verdict}")
        self.assertTrue(verdict.get("passes_tau"), "Backdoored model did not pass tau-ASR.")
        self.assertTrue(verdict.get("beats_clean_baseline"), "Backdoored model did not exceed clean baseline margin.")
        self.assertLess(float(clean_summary["max_asr"]), float(summary["tau_asr"]),
                        "Clean control unexpectedly met tau-ASR; investigate false-positive behavior.")


if __name__ == "__main__":
    unittest.main(verbosity=0)
