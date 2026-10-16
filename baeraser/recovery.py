#!/usr/bin/env python3
"""
Simplified BAERASER-inspired trigger recovery for the project described in
"Verifiable Federated Unlearning for Backdoor Removal".

IMPORTANT:
- This is NOT the original BAERASER implementation.
- It implements the project's Section 8 protocol:
  known 6x6 support; unknown position, RGB pattern, and target label;
  direct optimization of a free 6x6x3 trigger tensor.
- MSA/MESA, MINE/mutual-information, and BAERASER's original
  gradient-ascent unlearning are intentionally NOT implemented.

The core interface is recover_triggers(model, verification_loader, config).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model.model import LightweightResNet18


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic convolution can be substantially slower on some GPUs.
    # We leave cuDNN benchmark enabled only when the user does not request
    # strict determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CIFAR-10 standard normalization. The recovery code optimizes the trigger
# directly in this normalized input space.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RecoveryConfig:
    image_size: int = 32
    trigger_size: int = 6
    num_classes: int = 10

    # Project protocol: enumerate all valid locations and all labels.
    enumerate_all_locations: bool = True

    # Optimization
    optimization_steps: int = 15
    learning_rate: float = 0.05
    adam_betas: Tuple[float, float] = (0.9, 0.999)

    # Number of clean images used while optimizing a candidate.
    # Final ASR is evaluated on the complete verification set.
    optimization_images: int = 32

    # Number of (location, label) candidates optimized simultaneously.
    candidate_chunk_size: int = 32

    # Recovery decision threshold. MUST be fixed before the experiment.
    tau_asr: float = 0.80

    # Residual-evidence margin over the clean-model recovery baseline.
    delta_baseline: float = 0.10

    # Keep this many highest-scoring candidates per target label in memory.
    top_k_per_label: int = 10

    # Trigger initialization.
    init: str = "uniform"  # uniform or mean

    # Optional random restarts. Keep 1 on Kaggle free tier.
    restarts: int = 1

    # Evaluate ASR on non-target clean images by default.
    exclude_target_class_from_asr: bool = True

    # Data loader
    verification_batch_size: int = 128
    num_workers: int = 2


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load either a raw state_dict or a checkpoint dict."""
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            # Treat a dict whose values are tensors as a state_dict.
            state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    # Remove DataParallel prefix if present.
    cleaned = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned, strict=True)
    return model


# ---------------------------------------------------------------------------
# CIFAR-10 data
# ---------------------------------------------------------------------------

def make_cifar10_loaders(
    root: str,
    optimization_images: int = 32,
    verification_images: Optional[int] = None,
    batch_size: int = 128,
    num_workers: int = 2,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_set = datasets.CIFAR10(
        root=root, train=False, download=False, transform=transform
    )

    # A deterministic subset is useful for reproducible recovery.
    n = len(test_set)
    if verification_images is None or verification_images >= n:
        indices = list(range(n))
    else:
        rng = np.random.default_rng(12345)
        indices = rng.choice(n, verification_images, replace=False).tolist()

    verification_set = Subset(test_set, indices)

    loader = DataLoader(
        verification_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# ---------------------------------------------------------------------------
# Trigger geometry
# ---------------------------------------------------------------------------

def all_locations(image_size: int, trigger_size: int) -> List[Tuple[int, int]]:
    max_u = image_size - trigger_size
    max_v = image_size - trigger_size
    return [(u, v) for u in range(max_u + 1) for v in range(max_v + 1)]


def normalized_input_bounds(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Valid normalized values corresponding to raw RGB [0,1]."""
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    lo = (0.0 - mean_t) / std_t
    hi = (1.0 - mean_t) / std_t
    return lo, hi


def clamp_trigger(
    trigger: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    return torch.maximum(torch.minimum(trigger, hi), lo)


def make_location_indices(
    locations: Sequence[Tuple[int, int]],
    image_size: int,
    trigger_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [N, trigger_size^2] flattened H*W indices."""
    indices = []
    W = image_size
    for u, v in locations:
        rows = torch.arange(u, u + trigger_size, device=device)
        cols = torch.arange(v, v + trigger_size, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        indices.append((rr * W + cc).reshape(-1))
    return torch.stack(indices, dim=0)


def apply_trigger_batch(
    images: torch.Tensor,
    triggers: torch.Tensor,
    locations: Sequence[Tuple[int, int]],
    image_size: int,
) -> torch.Tensor:
    """
    Differentiably apply a different trigger to each candidate.

    images:   [B,3,H,W]
    triggers:  [C,3,S,S]
    returns:  [C,B,3,H,W]
    """
    C, _, S, _ = triggers.shape
    B = images.shape[0]

    idx = make_location_indices(
        locations, image_size=image_size, trigger_size=S, device=images.device
    )  # [C,S*S]

    idx3 = idx[:, None, :].expand(C, 3, S * S)

    canvas = torch.zeros(
        C, 3, image_size * image_size,
        device=images.device,
        dtype=images.dtype,
    )
    canvas = canvas.scatter(
        2, idx3, triggers.reshape(C, 3, S * S)
    ).view(C, 3, image_size, image_size)

    # Build the mask in flattened [C, 3, H*W] form so its dimensions
    # match idx3, then reshape it to [C, 3, H, W].
    mask = torch.zeros(
        C, 3, image_size * image_size,
        device=images.device,
        dtype=images.dtype,
    )
    mask = mask.scatter(
        2, idx3, torch.ones_like(triggers.reshape(C, 3, S * S))
    ).view(C, 3, image_size, image_size)

    return images.unsqueeze(0) * (1.0 - mask.unsqueeze(1)) + canvas.unsqueeze(1)


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def freeze_model(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def collect_optimization_batch(
    loader: DataLoader,
    n: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    total = 0

    for x, y in loader:
        take = min(n - total, x.size(0))
        xs.append(x[:take])
        ys.append(y[:take])
        total += take
        if total >= n:
            break

    if not xs:
        raise RuntimeError("Verification loader returned no samples.")

    x = torch.cat(xs, dim=0).to(device, non_blocking=True)
    y = torch.cat(ys, dim=0).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds), torch.cat(labels)


@torch.no_grad()
def clean_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return float(correct / max(total, 1))


@torch.no_grad()
def evaluate_trigger_chunk(
    model: nn.Module,
    loader: DataLoader,
    triggers: torch.Tensor,
    locations: Sequence[Tuple[int, int]],
    target_label: int,
    device: torch.device,
    image_size: int,
    exclude_target_class: bool = True,
) -> List[Dict[str, float]]:
    """
    Evaluate a whole optimized candidate chunk in parallel.

    This is substantially faster than running a separate full verification
    pass for every candidate. `triggers` is [C,3,S,S].
    """
    model.eval()
    triggers = triggers.to(device)

    C = triggers.size(0)
    success = torch.zeros(C, dtype=torch.long, device=device)
    total = torch.zeros(C, dtype=torch.long, device=device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        triggered = apply_trigger_batch(
            x, triggers, locations, image_size
        )  # [C,B,3,H,W]

        B = x.size(0)
        logits = model(
            triggered.reshape(C * B, 3, image_size, image_size)
        )
        pred = logits.argmax(dim=1).view(C, B)

        if exclude_target_class:
            mask = (y != target_label).unsqueeze(0).expand(C, -1)
            success += ((pred == target_label) & mask).sum(dim=1)
            total += mask.sum(dim=1)
        else:
            success += (pred == target_label).sum(dim=1)
            total += B

    asr = (success.float() / total.clamp_min(1).float()).cpu().tolist()

    return [
        {
            "asr": float(asr[i]),
            "evaluated_samples": int(total[i].item()),
        }
        for i in range(C)
    ]


def _initialize_triggers(
    count: int,
    config: RecoveryConfig,
    lo: torch.Tensor,
    hi: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if config.init == "mean":
        # Raw RGB 0.5 corresponds to (0.5-mean)/std in normalized space.
        mean = torch.tensor(CIFAR10_MEAN, device=device).view(3, 1, 1)
        std = torch.tensor(CIFAR10_STD, device=device).view(3, 1, 1)
        value = (0.5 - mean) / std
        return value.expand(count, -1, config.trigger_size, config.trigger_size).clone()

    if config.init == "uniform":
        return lo.unsqueeze(0) + torch.rand(
            count, 3, config.trigger_size, config.trigger_size, device=device
        ) * (hi - lo).unsqueeze(0)

    raise ValueError(f"Unknown init: {config.init}")


def optimize_candidate_chunk(
    model: nn.Module,
    base_images: torch.Tensor,
    target_label: int,
    locations: Sequence[Tuple[int, int]],
    config: RecoveryConfig,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """Optimize one chunk of independent triggers in parallel."""
    C = len(locations)
    trigger = nn.Parameter(
        _initialize_triggers(C, config, lo, hi, base_images.device)
    )

    optimizer = torch.optim.Adam(
        [trigger], lr=config.learning_rate, betas=config.adam_betas
    )

    target = torch.full(
        (C * base_images.size(0),),
        int(target_label),
        dtype=torch.long,
        device=base_images.device,
    )

    for _ in range(config.optimization_steps):
        optimizer.zero_grad(set_to_none=True)

        triggered = apply_trigger_batch(
            base_images, trigger, locations, config.image_size
        )
        # [C,B,3,H,W] -> [C*B,3,H,W]
        xb = triggered.reshape(
            C * base_images.size(0), 3, config.image_size, config.image_size
        )

        logits = model(xb)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            trigger.copy_(clamp_trigger(trigger, lo, hi))

    return trigger.detach()


def recover_triggers(
    model: nn.Module,
    verification_loader: DataLoader,
    config: Optional[RecoveryConfig] = None,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Enumerate all 729 positions x 10 labels for CIFAR-10 by default.

    Returns a list sorted by full-verification ASR.
    """
    config = config or RecoveryConfig()
    device = device or get_device()

    freeze_model(model)

    opt_x, opt_y = collect_optimization_batch(
        verification_loader,
        n=config.optimization_images,
        device=device,
    )

    locations = all_locations(config.image_size, config.trigger_size)
    lo, hi = normalized_input_bounds(device=device)

    if len(locations) != 729 and config.enumerate_all_locations:
        raise AssertionError(
            f"Expected 729 locations for 32x32 and 6x6, got {len(locations)}."
        )

    all_results: List[Dict] = []
    start = time.time()
    # Compute once, not once per candidate chunk.
    model_clean_acc = clean_accuracy(model, verification_loader, device)

    # For every candidate target label, optimize all locations.
    for target_label in range(config.num_classes):
        print(f"\nTarget label {target_label}/{config.num_classes - 1}")

        for start_idx in range(0, len(locations), config.candidate_chunk_size):
            loc_chunk = locations[
                start_idx:start_idx + config.candidate_chunk_size
            ]

            triggers = optimize_candidate_chunk(
                model=model,
                base_images=opt_x,
                target_label=target_label,
                locations=loc_chunk,
                config=config,
                lo=lo,
                hi=hi,
            )

            # Full verification evaluation. Do not use the optimization
            # subset for the final score. Evaluate the whole chunk in
            # parallel to keep Kaggle runtime practical.
            metrics_list = evaluate_trigger_chunk(
                model=model,
                loader=verification_loader,
                triggers=triggers,
                locations=loc_chunk,
                target_label=target_label,
                device=device,
                image_size=config.image_size,
                exclude_target_class=config.exclude_target_class_from_asr,
            )
            for i, loc in enumerate(loc_chunk):
                metrics = metrics_list[i]
                all_results.append({
                    "target_label": int(target_label),
                    "u": int(loc[0]),
                    "v": int(loc[1]),
                    "asr": metrics["asr"],
                    "clean_accuracy": model_clean_acc,
                    "evaluated_samples": metrics["evaluated_samples"],
                    "trigger": triggers[i].cpu(),
                })

            done = min(start_idx + len(loc_chunk), len(locations))
            print(
                f"  locations {done:3d}/{len(locations)} | "
                f"elapsed {(time.time()-start)/60:.1f} min"
            )

    all_results.sort(key=lambda r: r["asr"], reverse=True)
    return all_results


# ---------------------------------------------------------------------------
# Verification / baseline comparison
# ---------------------------------------------------------------------------

def summarize_recovery(
    results: List[Dict],
    tau_asr: float,
) -> Dict:
    if not results:
        return {
            "max_asr": 0.0,
            "num_candidates": 0,
            "num_above_tau": 0,
        }

    above = [r for r in results if r["asr"] >= tau_asr]
    best = results[0]

    return {
        "max_asr": float(best["asr"]),
        "best_target_label": int(best["target_label"]),
        "best_u": int(best["u"]),
        "best_v": int(best["v"]),
        "best_clean_accuracy": float(best["clean_accuracy"]),
        "num_candidates": len(results),
        "num_above_tau": len(above),
        "tau_asr": float(tau_asr),
    }


def classify_verification(
    post_summary: Dict,
    baseline_summary: Dict,
    delta_baseline: float,
) -> Dict:
    post_max = post_summary["max_asr"]
    base_max = baseline_summary["max_asr"]

    recovered = post_max >= post_summary["tau_asr"]
    beats_baseline = post_max >= base_max + delta_baseline

    if recovered and beats_baseline:
        status = "RESIDUAL_BACKDOOR_EVIDENCE"
    elif not recovered:
        status = "NO_RECOVERABLE_TRIGGER_UNDER_PROTOCOL"
    else:
        status = "INCONCLUSIVE_OR_BASELINE-COMPATIBLE"

    return {
        "status": status,
        "post_unlearning_max_asr": float(post_max),
        "clean_baseline_max_asr": float(base_max),
        "delta_baseline": float(delta_baseline),
        "passes_tau": bool(recovered),
        "beats_clean_baseline": bool(beats_baseline),
    }


# ---------------------------------------------------------------------------
# Ground-truth offline evaluation
# ---------------------------------------------------------------------------

def pattern_similarity(
    recovered: torch.Tensor,
    ground_truth: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> float:
    """
    1 - normalized MAE, averaged over channels.

    This metric is only for offline evaluation against the hidden ground truth.
    It is NEVER used during recovery.
    """
    r = recovered.float()
    g = ground_truth.float().to(r.device)

    denom = (hi - lo).to(r.device)
    mae = ((r - g).abs() / denom).mean().item()
    return float(max(0.0, 1.0 - mae))


def position_error(
    recovered_uv: Tuple[int, int],
    ground_truth_uv: Tuple[int, int],
) -> float:
    return float(math.sqrt(
        (recovered_uv[0] - ground_truth_uv[0]) ** 2 +
        (recovered_uv[1] - ground_truth_uv[1]) ** 2
    ))


def save_results(results: List[Dict], path: str) -> None:
    serializable = []
    for r in results:
        item = {k: v for k, v in r.items() if k != "trigger"}
        item["trigger_file_index"] = len(serializable)
        serializable.append(item)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_top_triggers(results: List[Dict], path: str, top_n: int = 50) -> None:
    top = results[:top_n]
    tensors = torch.stack([r["trigger"] for r in top], dim=0)
    metadata = []
    for r in top:
        metadata.append({
            "target_label": r["target_label"],
            "u": r["u"],
            "v": r["v"],
            "asr": r["asr"],
            "clean_accuracy": r["clean_accuracy"],
        })

    torch.save({
        "triggers": tensors,
        "metadata": metadata,
        "note": (
            "Triggers are in CIFAR-10 normalized input space. "
            "They were recovered without access to ground truth."
        ),
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Post-unlearning θU checkpoint.")
    parser.add_argument("--clean-model", default=None,
                        help="Clean-model checkpoint used for recovery baseline.")
    parser.add_argument(
        "--data-root",
        required=True,
        help="CIFAR-10 root directory containing cifar-10-batches-py/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where recovery results and summaries will be saved.",
    )
    parser.add_argument("--optimization-steps", type=int, default=15)
    parser.add_argument("--optimization-images", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tau-asr", type=float, default=0.80)
    parser.add_argument("--delta-baseline", type=float, default=0.10)
    parser.add_argument("--verification-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        parser.error(f"Model checkpoint not found: {args.model}")
    if args.clean_model is not None and not os.path.isfile(args.clean_model):
        parser.error(f"Clean-model checkpoint not found: {args.clean_model}")
    if not os.path.isdir(args.data_root):
        parser.error(f"CIFAR-10 data directory not found: {args.data_root}")
    expected_batch_dir = os.path.join(args.data_root, "cifar-10-batches-py")
    if not os.path.isdir(expected_batch_dir):
        parser.error(
            "Expected 'cifar-10-batches-py/' inside --data-root. "
            "Pass the parent directory, not the inner batch directory."
        )

    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    config = RecoveryConfig(
        optimization_steps=args.optimization_steps,
        optimization_images=args.optimization_images,
        candidate_chunk_size=args.candidate_chunk_size,
        learning_rate=args.lr,
        tau_asr=args.tau_asr,
        delta_baseline=args.delta_baseline,
        verification_batch_size=128,
        num_workers=args.num_workers,
    )

    loader = make_cifar10_loaders(
        root=args.data_root,
        optimization_images=config.optimization_images,
        verification_images=args.verification_images,
        batch_size=config.verification_batch_size,
        num_workers=config.num_workers,
    )

    # Post-unlearning model θU
    print("\n=== Recovering on post-unlearning model θU ===")
    model_u = LightweightResNet18(num_classes=config.num_classes).to(device)
    load_checkpoint(model_u, args.model, device)

    t0 = time.time()
    results_u = recover_triggers(model_u, loader, config, device)
    summary_u = summarize_recovery(results_u, config.tau_asr)

    save_results(results_u, str(out / "post_unlearning_results.json"))
    save_top_triggers(results_u, str(out / "post_unlearning_top_triggers.pt"))

    with open(out / "post_unlearning_summary.json", "w") as f:
        json.dump(summary_u, f, indent=2)

    print("\nPost-unlearning summary:")
    print(json.dumps(summary_u, indent=2))
    print(f"Recovery runtime: {(time.time()-t0)/60:.2f} min")

    # Clean baseline, if supplied.
    if args.clean_model:
        print("\n=== Recovering on clean-model baseline ===")
        clean = LightweightResNet18(num_classes=config.num_classes).to(device)
        load_checkpoint(clean, args.clean_model, device)

        t1 = time.time()
        results_clean = recover_triggers(clean, loader, config, device)
        summary_clean = summarize_recovery(results_clean, config.tau_asr)

        save_results(results_clean, str(out / "clean_baseline_results.json"))
        save_top_triggers(results_clean, str(out / "clean_baseline_top_triggers.pt"))

        verdict = classify_verification(
            summary_u, summary_clean, config.delta_baseline
        )

        with open(out / "clean_baseline_summary.json", "w") as f:
            json.dump(summary_clean, f, indent=2)

        with open(out / "verification_verdict.json", "w") as f:
            json.dump(verdict, f, indent=2)

        print("\nClean baseline summary:")
        print(json.dumps(summary_clean, indent=2))
        print("\nVerification verdict:")
        print(json.dumps(verdict, indent=2))
        print(f"Baseline recovery runtime: {(time.time()-t1)/60:.2f} min")
    else:
        print(
            "\nNo --clean-model supplied. The recovered ASR can be reported, "
            "but the project's baseline-margin decision cannot be completed."
        )

    with open(out / "recovery_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)


if __name__ == "__main__":
    main()
