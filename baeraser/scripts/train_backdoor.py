#!/usr/bin/env python3
"""Train a CIFAR-10 backdoored model using LightweightResNet18.

The trigger is a randomized 6x6 RGB patch in raw [0,1] pixel space.
Training is centralized only to create a controlled test checkpoint for the
BAERASER-inspired recovery module; it is NOT federated training or Wu unlearning.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from model import LightweightResNet18, count_parameters


MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_trigger(seed, size=6):
    """Generate a reproducible RGB trigger and its location and target label."""
    generator = torch.Generator().manual_seed(seed)

    # Trigger pixels are in raw image space [0, 1].
    patch = torch.rand((3, size, size), generator=generator)

    py = random.Random(seed + 1).randint(0, 32 - size)
    px = random.Random(seed + 2).randint(0, 32 - size)
    target = random.Random(seed + 3).randint(0, 9)

    return patch, (py, px), target


class PoisonedCIFAR(Dataset):
    """CIFAR-10 dataset with a fixed trigger applied to selected train images."""

    def __init__(
        self,
        root,
        train,
        poison_rate,
        patch,
        position,
        target,
        seed=42,
    ):
        self.ds = datasets.CIFAR10(
            root=root,
            train=train,
            download=False,
            transform=transforms.ToTensor(),
        )

        # Only the training split is poisoned.
        self.poison_rate = poison_rate if train else 0.0
        self.patch = patch
        self.position = position
        self.target = target

        rng = np.random.default_rng(seed + (0 if train else 100))

        # Do not poison images already belonging to the target class.
        eligible = [
            i
            for i, label in enumerate(self.ds.targets)
            if label != target
        ]

        n_poison = int(len(eligible) * self.poison_rate)

        if n_poison:
            selected = rng.choice(
                eligible,
                size=n_poison,
                replace=False,
            )
            self.poison_indices = set(selected.tolist())
        else:
            self.poison_indices = set()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]
        poisoned = idx in self.poison_indices

        if poisoned:
            py, px = self.position
            size = self.patch.shape[-1]

            image = image.clone()
            image[:, py:py + size, px:px + size] = self.patch
            label = self.target

        # Normalize after inserting the trigger in raw pixel space.
        image = (image - MEAN) / STD

        return image, int(label), bool(poisoned)


@torch.no_grad()
def evaluate(model, loader, device, patch, position, target):
    """Evaluate clean accuracy and attack success rate on triggered test images."""
    model.eval()

    correct = 0
    total = 0
    asr_success = 0
    asr_total = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        predictions = model(x).argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.numel()

        # ASR is evaluated only on clean test examples not in the target class.
        non_target = y != target

        if non_target.any():
            triggered_x = x[non_target].clone()

            py, px = position
            size = patch.shape[-1]

            # Replace the normalized image patch with the normalized trigger.
            normalized_patch = (
                patch.to(device) - MEAN.to(device)
            ) / STD.to(device)

            triggered_x[
                :, :, py:py + size, px:px + size
            ] = normalized_patch.unsqueeze(0)

            triggered_predictions = model(triggered_x).argmax(dim=1)

            asr_success += (
                triggered_predictions == target
            ).sum().item()

            asr_total += triggered_predictions.numel()

    return {
        "clean_accuracy": correct / max(total, 1),
        "trigger_asr": asr_success / max(asr_total, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a backdoored CIFAR-10 LightweightResNet18 model."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "Path to the CIFAR-10 root directory containing "
            "cifar-10-batches-py/"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/kaggle/working/backdoored_model.pt",
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.10,
        help="Fraction of eligible training images to poison.",
    )
    parser.add_argument("--trigger-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    if not 0 <= args.poison_rate <= 1:
        raise ValueError("--poison-rate must be between 0 and 1.")

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if not 1 <= args.trigger_size <= 32:
        raise ValueError("--trigger-size must be between 1 and 32.")

    if args.trigger_size != 6:
        print(
            "WARNING: Recovery assumes a 6x6 trigger. "
            "Use --trigger-size 6 for protocol consistency."
        )

    if not os.path.isdir(args.data):
        raise FileNotFoundError(
            f"Dataset directory does not exist: {args.data}"
        )

    expected_batch_dir = os.path.join(
        args.data, "cifar-10-batches-py"
    )

    if not os.path.isdir(expected_batch_dir):
        raise FileNotFoundError(
            "Could not find 'cifar-10-batches-py' inside the supplied "
            f"dataset root: {args.data}\n"
            "Pass the directory containing cifar-10-batches-py, "
            "not the batch directory itself."
        )

    seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    patch, position, target = make_trigger(
        args.seed, args.trigger_size
    )

    train_ds = PoisonedCIFAR(
        root=args.data,
        train=True,
        poison_rate=args.poison_rate,
        patch=patch,
        position=position,
        target=target,
        seed=args.seed,
    )

    test_ds = PoisonedCIFAR(
        root=args.data,
        train=False,
        poison_rate=0.0,
        patch=patch,
        position=position,
        target=target,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = LightweightResNet18().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    criterion = nn.CrossEntropyLoss()

    print(f"Dataset: {args.data}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Trigger position (y, x): {position}")
    print(f"Target label: {target}")
    print(f"Poison rate: {args.poison_rate}")
    print(f"Poisoned training images: {len(train_ds.poison_indices)}")

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        metrics = evaluate(
            model,
            test_loader,
            device,
            patch,
            position,
            target,
        )

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            **metrics,
        }

        history.append(row)
        print(json.dumps(row))

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "model_name": "LightweightResNet18",
        "num_classes": 10,
        "normalization": {
            "mean": MEAN.flatten().tolist(),
            "std": STD.flatten().tolist(),
        },
        # Ground truth for offline evaluation only.
        # Do not expose this metadata to the recovery algorithm.
        "trigger_metadata": {
            "size": args.trigger_size,
            "position": list(position),
            "target_label": target,
            "pattern_raw_0_1": patch.tolist(),
            "poison_rate": args.poison_rate,
            "seed": args.seed,
        },
        "history": history,
    }

    torch.save(checkpoint, args.out)

    metadata_path = (
        os.path.splitext(args.out)[0] + "_metadata.json"
    )

    with open(metadata_path, "w") as metadata_file:
        json.dump(
            {
                "model_name": "LightweightResNet18",
                "trigger_size": args.trigger_size,
                "position": list(position),
                "target_label": target,
                "poison_rate": args.poison_rate,
                "seed": args.seed,
                "history": history,
            },
            metadata_file,
            indent=2,
        )

    print(f"Saved checkpoint: {args.out}")
    print(f"Saved metadata: {metadata_path}")
    print(
        "This is a centralized backdoor test model, not a federated "
        "or post-Wu-unlearning model."
    )


if __name__ == "__main__":
    main()
