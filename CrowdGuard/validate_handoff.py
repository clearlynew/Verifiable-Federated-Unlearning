"""Audit a CrowdGuard handoff file (schema version 2).

    python validate_handoff.py outputs/crowdguard_handoff.pt

Checks (exit code 1 if any ERROR):
  * required fields, round count, all clients present in every round
  * weight bookkeeping: theta_0 / theta_k / round chain, and
        G_after = G_before + mean_i(delta_i)   for every round
        theta_k = theta_0 + (1/N) * sum_t sum_i delta_i^t
    (the identity the Wu historical-update subtraction relies on)
  * commit-reveal: the commitment digest matches the commitment list, every
    commitment is recomputed from its reveal, recorded statuses must match,
    and only valid reveals appear in `votes`
  * the per-round detections M_t are recomputed from the revealed votes with
    the same stacked-clustering code, and must equal the recorded ones
  * final M and detection counts match the recorded rule / tau
  * client partitions and the KD / evaluation test split are disjoint and
    correctly sized
"""

import argparse
import sys

import numpy as np

import commit_reveal as cr
from handoff_utils import detection_counts, select_final_m, stacked_clustering_vote

REQUIRED_KEYS = (
    "schema_version", "theta_0", "theta_k", "round_history", "final_M",
    "final_M_rule", "detection_counts", "ground_truth_malicious_clients",
    "config", "trigger", "client_train_indices", "test_split", "environment",
)
NUM_TRAIN_IMAGES = 50000
NUM_TEST_IMAGES = 10000


def _np(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _float_keys(state):
    return [k for k, v in state.items() if np.issubdtype(_np(v).dtype, np.floating)]


def _max_rel_err(a, b):
    a = _np(a).astype(np.float64)
    b = _np(b).astype(np.float64)
    scale = max(1.0, float(np.abs(b).max()) if b.size else 1.0)
    return float(np.abs(a - b).max()) / scale if a.size else 0.0


def _state_gap(a, b, keys):
    return max((_max_rel_err(a[k], b[k]) for k in keys), default=0.0)


def validate(handoff, rtol=1e-5):
    """Return dict(errors=[...], warnings=[...], notes=[...])."""
    errors, warnings, notes = [], [], []

    def err(msg):
        errors.append(msg)

    missing = [k for k in REQUIRED_KEYS if k not in handoff]
    if missing:
        err(f"missing required fields: {missing}")
        return {"errors": errors, "warnings": warnings, "notes": notes}
    if handoff["schema_version"] != 2:
        err(f"unsupported schema_version {handoff['schema_version']!r} (expected 2)")
        return {"errors": errors, "warnings": warnings, "notes": notes}

    config = handoff["config"]
    history = handoff["round_history"]
    n_clients = config["num_clients"]
    context = cr.make_context(config["seed"], n_clients, config["pmr"])

    if config.get("tamper_test"):
        warnings.append("config.tamper_test is True: this run is a self-test, "
                        "its results must not be used")
    if len(history) != config["rounds"]:
        err(f"{len(history)} rounds recorded but config.rounds = {config['rounds']}")

    names = sorted(history[0]["updates"].keys()) if history else []
    if len(names) != n_clients:
        err(f"round 0 has {len(names)} clients, config.num_clients = {n_clients}")
    if any(("malicious" in n or "benign" in n) for n in names):
        notes.append("client IDs reveal ground truth (benign_*/malicious_*); fine for "
                     "internal use, anonymize before a blind downstream stage")
    truth = set(handoff["ground_truth_malicious_clients"])
    if not truth <= set(names):
        err(f"ground truth {sorted(truth - set(names))} is not among the clients")

    # ---- per-round checks ---------------------------------------------------
    float_keys = _float_keys(handoff["theta_0"])
    structure_ok = True
    for i, rec in enumerate(history):
        tag = f"round {i}"
        if rec.get("round") != i:
            err(f"{tag}: record says round={rec.get('round')}")
        if sorted(rec["updates"].keys()) != names:
            err(f"{tag}: client set differs from round 0")
            structure_ok = False
            continue
        if sorted(rec["attack_active"].keys()) != names:
            err(f"{tag}: attack_active does not cover every client")

        cr_rec = rec.get("commit_reveal")
        if cr_rec is None:
            err(f"{tag}: no commit_reveal record")
            continue
        if cr_rec["context"] != context:
            err(f"{tag}: commitment context {cr_rec['context']!r} != expected {context!r}")
        if sorted(cr_rec["commitments"].keys()) != names:
            warnings.append(f"{tag}: commitments missing for "
                            f"{sorted(set(names) - set(cr_rec['commitments']))}")
        digest = cr.commitment_digest(cr_rec["commitments"], i, context)
        if cr_rec.get("commitment_digest") != digest:
            err(f"{tag}: commitment digest does not match the commitment list")
        not_confirmed = [v for v, ok in cr_rec.get("own_commitment_confirmed", {}).items()
                         if not ok]
        if not_confirmed:
            warnings.append(f"{tag}: validators {not_confirmed} did not find their own "
                            f"commitment published unchanged and refused to reveal")
        status, valid = cr.verify_round(
            cr_rec["commitments"], cr_rec["reveals"], names, i, context)
        if status != cr_rec["status"]:
            diff = {v: (cr_rec["status"].get(v), status[v])
                    for v in names if cr_rec["status"].get(v) != status[v]}
            err(f"{tag}: recorded reveal status disagrees with recomputation "
                f"(recorded, recomputed): {diff}")
        if valid != cr_rec["valid_validators"]:
            err(f"{tag}: valid_validators mismatch")
        bad = {v: s for v, s in status.items() if s != cr.STATUS_VALID}
        if bad:
            warnings.append(f"{tag}: {len(bad)} reveal(s) rejected: {bad}")

        # votes must be exactly the valid reveals
        if sorted(rec["votes"].keys()) != sorted(valid):
            err(f"{tag}: `votes` rows are not exactly the valid validators")
        else:
            for v in valid:
                recorded = [rec["votes"][v][c] for c in names]
                if recorded != list(cr_rec["reveals"][v]["votes"]):
                    err(f"{tag}: `votes` for {v} differ from its reveal")

        # independent recomputation of M_t from the revealed votes
        rows = [cr_rec["reveals"][v]["votes"] for v in valid]
        final = stacked_clustering_vote(rows, len(names))
        recomputed = sorted(n for n, x in zip(names, final) if x == 0)
        if recomputed != sorted(set(rec["detected_clients"])):
            err(f"{tag}: recorded detections {sorted(rec['detected_clients'])} != "
                f"recomputed from revealed votes {recomputed}")

        # weight bookkeeping: after = before + mean(updates)
        gap = 0.0
        for k in float_keys:
            mean_update = sum(_np(rec["updates"][n][k]).astype(np.float64)
                              for n in names) / len(names)
            expect = _np(rec["global_model_before"][k]).astype(np.float64) + mean_update
            gap = max(gap, _max_rel_err(expect, rec["global_model_after"][k]))
        if gap > rtol:
            err(f"{tag}: global_after != global_before + mean(updates) "
                f"(max rel err {gap:.2e} > {rtol:.0e})")

    # ---- model chain ----------------------------------------------------------
    if history:
        g = _state_gap(handoff["theta_0"], history[0]["global_model_before"], float_keys)
        if g > rtol:
            err(f"theta_0 != round 0 global_model_before (rel err {g:.2e})")
        g = _state_gap(handoff["theta_k"], history[-1]["global_model_after"], float_keys)
        if g > rtol:
            err(f"theta_k != last round global_model_after (rel err {g:.2e})")
        for i in range(len(history) - 1):
            g = _state_gap(history[i]["global_model_after"],
                           history[i + 1]["global_model_before"], float_keys)
            if g > rtol:
                err(f"round {i} after != round {i + 1} before (rel err {g:.2e})")

        # theta_k = theta_0 + (1/N) * sum of all deltas
        if structure_ok:
            worst = 0.0
            for k in float_keys:
                total = sum(_np(r["updates"][n][k]).astype(np.float64)
                            for r in history for n in names) / len(names)
                expect = _np(handoff["theta_0"][k]).astype(np.float64) + total
                worst = max(worst, _max_rel_err(expect, handoff["theta_k"][k]))
            notes.append(f"theta_k vs theta_0 + (1/N)*sum(deltas): rel err {worst:.2e}")
            # float32 rounding accumulates over rounds, so allow a little slack
            if worst > rtol * max(1, len(history)):
                err(f"theta_k != theta_0 + (1/N)*sum(deltas) (rel err {worst:.2e})")
        else:
            notes.append("skipped the theta_k identity check: a round has a broken client set")

    # ---- final M and counts -----------------------------------------------------
    rule = handoff["final_M_rule"]
    expected_m = select_final_m(history, rule["rule"], rule["tau"])
    if sorted(handoff["final_M"]) != sorted(expected_m):
        err(f"final_M {sorted(handoff['final_M'])} != recomputed {expected_m} "
            f"for rule {rule['rule']} tau {rule['tau']}")
    if handoff["detection_counts"] != detection_counts(history):
        err("detection_counts do not match the per-round detections")
    if rule.get("rounds") != len(history):
        err("final_M_rule.rounds != number of rounds")

    # ---- partitions and test split -------------------------------------------------
    parts = handoff["client_train_indices"]
    if not isinstance(parts, dict) or sorted(parts.keys()) != names:
        err("client_train_indices must have one entry per client")
    else:
        flat = [i for n in names for i in parts[n]]
        partition_type = config.get("partition", "iid")
        samples_per_client = config.get("samples_per_client")
        
        # 1. Total dataset / partition size check
        expected_total = config.get("total_train_samples")
        if expected_total is None:
            expected_total = samples_per_client * n_clients if samples_per_client else NUM_TRAIN_IMAGES        
        if len(flat) != expected_total:
            err(f"total partitioned samples ({len(flat)}) != expected total ({expected_total})")

        # 2. Per-client size checks based on partition rule
        if partition_type == "iid":
            if samples_per_client and any(len(parts[n]) != samples_per_client for n in names):
                err("a client partition has the wrong size for IID mode")
        elif partition_type == "dirichlet":
            if any(len(parts[n]) == 0 for n in names):
                err("a client partition has 0 samples under Dirichlet distribution")

        # 3. Disjointness and range checks
        if len(set(flat)) != len(flat):
            err("client partitions overlap")
        if flat and (min(flat) < 0 or max(flat) >= NUM_TRAIN_IMAGES):
            err("client partition index out of range")

    split = handoff["test_split"]
    kd, ev = split["kd_indices"], split["eval_indices"]
    if len(kd) != config["kd_reference_size"]:
        err(f"kd split has {len(kd)} images, expected {config['kd_reference_size']}")
    if set(kd) & set(ev):
        err("KD and evaluation test splits overlap")
    if set(kd) | set(ev) != set(range(NUM_TEST_IMAGES)):
        err(f"KD + evaluation splits do not cover range [0, {NUM_TEST_IMAGES}) exactly")

    return {"errors": errors, "warnings": warnings, "notes": notes}


def load_handoff(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("handoff")
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()

    result = validate(load_handoff(args.handoff), rtol=args.rtol)
    for level in ("notes", "warnings", "errors"):
        for message in result[level]:
            print(f"{level[:-1].upper():8s} {message}")
    if result["errors"]:
        print(f"\nFAILED: {len(result['errors'])} error(s)")
        sys.exit(1)
    print("\nPASSED: handoff is internally consistent")


if __name__ == "__main__":
    main()
