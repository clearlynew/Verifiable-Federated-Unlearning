#!/usr/bin/env python
# coding: utf-8

"""CrowdGuard experiment runner for the verifiable federated-unlearning project.

The repository's original CrowdGuard HLBIM + stacked-clustering detector is
retained.  The project modification is detection-only operation: detected
clients are recorded but are NOT filtered from equal-weight FedAvg, because the
next Wu stage requires their historical submitted updates.
"""

import argparse
import copy
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms

from CrowdGuardClientValidation import CrowdGuardClientValidation
from adaptive_backdoor import adaptive_train_and_scale, sample_trigger, apply_trigger
from experiment_config import ExperimentConfig
from lightweight_resnet18 import LightweightResNet18, count_parameters
from commit_reveal import (
    commitment_digest, make_commitment, make_context, new_nonce, tamper_reveal,
    verify_round,
)
from handoff_utils import (
    atomic_torch_save, collect_environment_metadata, detection_counts,
    select_final_m, split_test_indices, stacked_clustering_vote,
)

from openfl.experimental.workflow.interface import Aggregator, Collaborator, FLSpec
from openfl.experimental.workflow.placement import aggregator, collaborator
from openfl.experimental.workflow.runtime import LocalRuntime

warnings.filterwarnings("ignore")

MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD_DEV = torch.tensor([0.2023, 0.1994, 0.2010])
VOTE_FOR_BENIGN = 1
VOTE_FOR_POISONED = 0
LOG_INTERVAL = 10

def partition_dirichlet(dataset, num_clients, alpha, num_classes=10, seed=10):
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    
    client_indices = [[] for _ in range(num_clients)]
    min_size = 0
    
    while min_size < 10:
        client_indices = [[] for _ in range(num_clients)]
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients), num_classes)
        
        for c in range(num_classes):
            idx_k = np.where(labels == c)[0]
            np.random.shuffle(idx_k)
            
            proportions_c = proportions[c]
            proportions_c = proportions_c / proportions_c.sum()
            splits = (np.cumsum(proportions_c) * len(idx_k)).astype(int)[:-1]
            
            idx_k_split = np.split(idx_k, splits)
            for i in range(num_clients):
                client_indices[i].extend(idx_k_split[i])
        
        min_size = min(len(idx) for idx in client_indices)

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        
    return [Subset(dataset, indices) for indices in client_indices]


# ---------------------------------------------------------------------------
# Reproducibility / model utilities
# ---------------------------------------------------------------------------

def seed_random_generators(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def default_optimizer(model, learning_rate, momentum=0.9, optimizer_type="SGD"):
    if optimizer_type.upper() == "SGD":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if optimizer_type.upper() == "ADAM":
        return optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def model_state_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def model_delta(local_model, global_state):
    delta = {}
    local_state = local_model.state_dict()
    for name, value in local_state.items():
        reference = global_state[name]
        if torch.is_floating_point(value):
            delta[name] = (value.detach().cpu() - reference.detach().cpu()).clone()
        else:
            # Integer buffers such as num_batches_tracked are not part of the
            # mathematical FedAvg update.  Retain them separately as metadata.
            delta[name] = torch.zeros_like(reference.detach().cpu())
    return delta


def apply_delta_to_state(global_state, delta):
    result = {}
    for name, value in global_state.items():
        if torch.is_floating_point(value):
            result[name] = value + delta[name]
        else:
            result[name] = value.clone()
    return result


def fed_avg(models):
    if not models:
        raise ValueError("FedAvg received no models")
    result = copy.deepcopy(models[0])
    state = result.state_dict()
    state_dicts = [model.state_dict() for model in models]
    for name in state:
        if torch.is_floating_point(state[name]):
            stacked = torch.stack([sd[name].detach().cpu() for sd in state_dicts])
            state[name] = stacked.mean(dim=0)
        else:
            state[name] = state_dicts[0][name].detach().cpu().clone()
    result.load_state_dict(state)
    return result


def evaluate(model, loader, device):
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item() * target.size(0)
            correct += output.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    model.cpu()
    return loss_sum / max(1, total), correct / max(1, total)


# ---------------------------------------------------------------------------
# CrowdGuard stacked clustering helper functions
# ---------------------------------------------------------------------------

def create_cluster_map_from_labels(expected_number_of_labels, clustering_labels):
    assert len(clustering_labels) == expected_number_of_labels
    clusters = {}
    for i, cluster in enumerate(clustering_labels):
        clusters.setdefault(cluster, []).append(i)
    return {index: np.array(cluster) for index, cluster in clusters.items()}


def determine_biggest_cluster(clustering):
    if not clustering:
        raise ValueError("Cannot choose a cluster from an empty clustering")
    return max(clustering, key=lambda cluster_id: len(clustering[cluster_id]))


# select_final_m (union / frequency rules) now lives in handoff_utils.py and is
# imported above; the rule and tau come from ExperimentConfig.


# ---------------------------------------------------------------------------
# OpenFL workflow
# ---------------------------------------------------------------------------

class FederatedFlow(FLSpec):
    def __init__(self, model, optimizer_templates, config, device="cpu",
                 run_info=None, **kwargs):
        super().__init__(**kwargs)
        # run_info: client partition indices and test split, saved into the handoff
        self.run_info = run_info or {}
        self.model = model
        self.global_model = copy.deepcopy(model)
        self.theta_0 = model_state_cpu(model)
        self.optimizer_templates = optimizer_templates
        self.config = config
        self.device = device
        self.round_num = 0
        self.round_history = []
        self.all_models = {}
        self.all_votes_by_name = {}
        self.start_time = None
        self.trigger = sample_trigger(
            config.trigger_size,
            target_label=config.target_label,
            seed=config.seed,
        )
        self.malicious_client_names = {
            f"malicious_{i:02d}" for i in range(config.num_malicious_clients)
        }

    @aggregator
    def start(self):
        self.start_time = time.time()
        self.collaborators = self.runtime.collaborators
        self.private = 10
        print("#" * 60)
        print("CrowdGuard project run")
        print(f"Clients: {self.config.num_clients}")
        print(f"Malicious clients: {sorted(self.malicious_client_names)}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Trigger: {self.trigger.size}x{self.trigger.size}, "
              f"position=({self.trigger.top},{self.trigger.left}), "
              f"target={self.trigger.target_label}")
        print("CrowdGuard mode: DETECTION ONLY (all models aggregated)")
        print("#" * 60)
        self.next(self.train, foreach="collaborators", exclude=["private"])

    @collaborator
    def train(self):
        self.collaborator_name = self.input
        global_state = model_state_cpu(self.global_model)
        self.model.load_state_dict(global_state)
        self.model.to(self.device)

        optimizer = default_optimizer(
            self.model,
            self.config.learning_rate,
            self.config.momentum,
            self.optimizer_templates[self.input],
        )

        is_malicious = self.collaborator_name in self.malicious_client_names
        attack_active = is_malicious and self.round_num >= self.config.attack_start_round

        if attack_active:
            scale_factor = self.config.num_clients / max(1, self.config.num_malicious_clients)
            attack_stats = adaptive_train_and_scale(
                self.model,
                global_state,
                self.train_loader,
                optimizer,
                self.device,
                self.trigger,
                poison_rate=self.config.poison_rate,
                alpha=self.config.alpha,
                local_epochs=self.config.local_epochs,
                scale_factor=scale_factor,
            )
            self.attack_stats = attack_stats
        else:
            criterion = nn.CrossEntropyLoss()
            losses = []
            self.model.train()
            for _ in range(self.config.local_epochs):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % LOG_INTERVAL == 0:
                        losses.append(float(loss.detach().cpu()))
            self.attack_stats = {"mean_loss": float(np.mean(losses)) if losses else 0.0,
                                 "scale_factor": 1.0}

        self.model.cpu()
        self.training_completed = True
        self.attack_active = attack_active
        self.next(self.collect_models, exclude=["training_completed"])

    @aggregator
    def collect_models(self, inputs):
        # This is the project handoff boundary: retain every submitted model
        # update before any CrowdGuard detection result is used.
        self.all_models = {
            item.collaborator_name: item.model.cpu() for item in inputs
        }
        global_state = model_state_cpu(self.global_model)

        updates = {
            name: model_delta(model, global_state)
            for name, model in self.all_models.items()
        }
        attack_status = {
            item.collaborator_name: bool(item.attack_active) for item in inputs
        }

        self.current_round_record = {
            "round": self.round_num,
            "global_model_before": global_state,
            "updates": updates,
            "attack_active": attack_status,
        }
        self.next(self.local_validation, foreach="collaborators")

    @collaborator
    def local_validation(self):
        self.collaborator_name = self.input
        all_names = sorted(self.all_models.keys())
        all_models = [self.all_models[name] for name in all_names]
        own_client_index = all_names.index(self.collaborator_name)

        detected = CrowdGuardClientValidation.validate_models(
            self.global_model,
            all_models,
            own_client_index,
            self.train_loader,
            self.device,
        )
        detected = sorted(detected)
        print(f"Round {self.round_num}: {self.collaborator_name} detected indices {detected}")

        votes = []
        for index in range(len(all_models)):
            if index == own_client_index:
                votes.append(VOTE_FOR_BENIGN)
            elif index in detected:
                votes.append(VOTE_FOR_POISONED)
            else:
                votes.append(VOTE_FOR_BENIGN)

        # OpenFL reuses each collaborator's state object across rounds, so clear last
        # round's reveal state explicitly: during the commit phase nothing from any
        # reveal may be visible to the aggregator.
        self.reveal_record = None
        self.own_commitment_confirmed = None

        # PHASE 1 (commit): publish only a hash of the vote vector.  The votes and
        # nonce stay in this validator's private vault.  `vote_vault` is an OpenFL
        # private attribute: it persists for this collaborator across steps but is
        # stripped before the state is sent to the aggregator.
        context = make_context(self.config.seed, self.config.num_clients, self.config.pmr)
        nonce = new_nonce()
        commitment = make_commitment(
            votes, self.round_num, self.collaborator_name, all_names, nonce, context)
        self.vote_vault[self.round_num] = {
            "votes": [int(v) for v in votes], "nonce": nonce, "commitment": commitment,
        }
        self.commit_record = {
            "validator": self.collaborator_name,
            "round": self.round_num,
            "commitment": commitment,
        }
        self.next(self.collect_commitments)

    @aggregator
    def collect_commitments(self, inputs):
        # The commit phase closes here: the aggregator holds every commitment and
        # has seen no vote.  The list (and its digest) is fixed before any reveal.
        context = make_context(self.config.seed, self.config.num_clients, self.config.pmr)
        self.round_commitments = {
            item.collaborator_name: item.commit_record["commitment"] for item in inputs
        }
        self.round_commitment_digest = commitment_digest(
            self.round_commitments, self.round_num, context)
        self.next(self.reveal_votes, foreach="collaborators")

    @collaborator
    def reveal_votes(self):
        # PHASE 2 (reveal).  First check the published list still contains this
        # validator's own commitment unchanged; if not, refuse to reveal.
        self.collaborator_name = self.input
        entry = self.vote_vault.pop(self.round_num)
        published = self.round_commitments.get(self.collaborator_name)
        self.own_commitment_confirmed = (published == entry["commitment"])
        if self.own_commitment_confirmed:
            self.reveal_record = {"votes": entry["votes"], "nonce": entry["nonce"]}
        else:
            print(f"Round {self.round_num}: {self.collaborator_name} sees a changed or "
                  f"missing commitment and refuses to reveal")
            self.reveal_record = None
        self.next(self.defend)

    @aggregator
    def defend(self, inputs):
        all_names = sorted(self.all_models.keys())
        context = make_context(self.config.seed, self.config.num_clients, self.config.pmr)

        # Commitments are the ones fixed in phase 1 (collect_commitments), before any
        # reveal existed.  Reveals arrive only now, from reveal_votes.
        commitments = dict(self.round_commitments)
        reveals = {
            item.collaborator_name:
                (dict(item.reveal_record) if item.reveal_record else None)
            for item in inputs
        }
        confirmed = {item.collaborator_name: bool(item.own_commitment_confirmed)
                     for item in inputs}
        if self.config.tamper_test and reveals.get(all_names[0]) is not None:
            # Self-test only: corrupt one validator's reveal; it must be rejected.
            reveals = tamper_reveal(reveals, all_names[0])

        status, valid_validators = verify_round(
            commitments, reveals, all_names, self.round_num, context)
        rejected = {v: why for v, why in status.items() if why != "valid"}
        if rejected:
            print(f"Round {self.round_num}: REJECTED reveals: {rejected}")

        # Only valid reveals enter the vote matrix (rows in a fixed order).
        binary_votes = [reveals[v]["votes"] for v in valid_validators]
        all_votes_by_name = {
            v: dict(zip(all_names, reveals[v]["votes"])) for v in valid_validators
        }

        # Original CrowdGuard stacked-clustering aggregation (Alg. 3); the code
        # lives in handoff_utils so the audit script recomputes it identically.
        final_voting = stacked_clustering_vote(binary_votes, len(all_names))

        detected_names = [
            name for name, vote in zip(all_names, final_voting)
            if vote == VOTE_FOR_POISONED
        ]
        detected_names = sorted(set(detected_names))

        # Project modification: DO NOT filter detected models.  Equal-weight
        # FedAvg receives every participating client update.
        aggregated_model = fed_avg([self.all_models[name] for name in all_names])
        self.model = aggregated_model
        self.global_model = copy.deepcopy(aggregated_model)

        self.current_round_record["votes"] = all_votes_by_name
        self.current_round_record["commit_reveal"] = {
            "context": context,
            "commitments": commitments,
            "commitment_digest": self.round_commitment_digest,
            "own_commitment_confirmed": confirmed,
            "reveals": reveals,
            "status": status,
            "valid_validators": valid_validators,
        }
        self.current_round_record["detected_clients"] = detected_names
        self.current_round_record["global_model_after"] = model_state_cpu(aggregated_model)
        self.round_history.append(self.current_round_record)

        # Crash insurance: each finished round is saved on its own, so a dead
        # session costs one round instead of the whole run.
        atomic_torch_save(
            self.current_round_record,
            os.path.join(self.config.output_dir, "rounds",
                         f"round_{self.current_round_record['round']:02d}.pt"),
        )

        self.round_num += 1
        if self.round_num < self.config.rounds:
            self.next(self.train, foreach="collaborators")
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        final_m = select_final_m(
            self.round_history, rule=self.config.final_m_rule, tau=self.config.final_m_tau)
        elapsed = time.time() - self.start_time
        handoff = {
            "schema_version": 2,
            "final_M_rule": {"rule": self.config.final_m_rule, "tau": self.config.final_m_tau,
                             "rounds": len(self.round_history)},
            "detection_counts": detection_counts(self.round_history),
            "client_train_indices": self.run_info.get("client_train_indices"),
            "test_split": self.run_info.get("test_split"),
            "environment": collect_environment_metadata(),
            "theta_0": self.theta_0,
            "theta_k": model_state_cpu(self.model),
            "round_history": self.round_history,
            "final_M": final_m,
            "ground_truth_malicious_clients": sorted(self.malicious_client_names),
            "config": self.config.to_dict(),
            "trigger": self.trigger.to_dict(),
            "elapsed_seconds": elapsed,
            "model_parameter_count": count_parameters(self.model),
        }

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, "crowdguard_handoff.pt")
        atomic_torch_save(handoff, output_path)

        metadata = {
            "final_M": final_m,
            "final_M_rule": handoff["final_M_rule"],
            "detection_counts": handoff["detection_counts"],
            "environment": handoff["environment"],
            "ground_truth_malicious_clients": sorted(self.malicious_client_names),
            "config": self.config.to_dict(),
            "trigger": {
                "size": self.trigger.size,
                "top": self.trigger.top,
                "left": self.trigger.left,
                "target_label": self.trigger.target_label,
                "pattern": self.trigger.pattern.tolist(),
            },
            "elapsed_seconds": elapsed,
            "model_parameter_count": count_parameters(self.model),
        }
        with open(os.path.join(self.config.output_dir, "crowdguard_metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2)

        print("#" * 60)
        print("CrowdGuard run completed")
        print(f"Final M: {final_m}")
        print(f"Handoff: {output_path}")
        print(f"Elapsed: {elapsed:.2f}s")
        print("#" * 60)


def build_datasetsIID(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD_DEV.tolist()),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                    transform=transform)

    if config.num_clients * config.samples_per_client > len(train_dataset):
        raise ValueError(
            f"Need {config.num_clients * config.samples_per_client} training samples "
            f"but CIFAR-10 has only {len(train_dataset)}."
        )

    # Deterministic IID partition of the 50,000 training samples.
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(train_dataset))
    indices = indices[:config.num_clients * config.samples_per_client]

    train_x = torch.stack([train_dataset[i][0] for i in indices])
    train_y = torch.tensor([train_dataset[i][1] for i in indices], dtype=torch.long)
    test_x = torch.stack([item[0] for item in test_dataset])
    test_y = torch.tensor([item[1] for item in test_dataset], dtype=torch.long)

    client_loaders = {}
    for client_id in range(config.num_clients):
        start = client_id * config.samples_per_client
        end = start + config.samples_per_client
        x = train_x[start:end]
        y = train_y[start:end]
        client_loaders[client_id] = DataLoader(
            TensorDataset(x, y), batch_size=config.batch_size, shuffle=True
        )

    # Test set split (fixed seed, stratified): kd_reference_size images are kept
    # aside as unlabeled KD reference data for the later Wu stage; the remaining
    # images are the held-out evaluation set.  Diagnostics and every reported
    # accuracy/ASR must use the held-out part only.
    kd_indices, eval_indices = split_test_indices(
        test_y.numpy(), config.kd_reference_size, config.test_split_seed)

    # Per-collaborator diagnostics use a bounded held-out subset; the complete
    # test set is intentionally not duplicated into every collaborator state.
    eval_count = min(1000, len(eval_indices))
    diag = torch.tensor(eval_indices[:eval_count], dtype=torch.long)
    clean_test_loader = DataLoader(
        TensorDataset(test_x[diag], test_y[diag]),
        batch_size=1000, shuffle=False
    )

    data_info = {
        "train_partition": {
            client_id: [int(i) for i in indices[
                client_id * config.samples_per_client:
                (client_id + 1) * config.samples_per_client]]
            for client_id in range(config.num_clients)
        },
        "test_split": {
            "seed": config.test_split_seed,
            "kd_reference_size": config.kd_reference_size,
            "kd_indices": kd_indices,
            "eval_indices": eval_indices,
        },
    }
    return client_loaders, clean_test_loader, data_info

def build_datasetsDirichlet(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD_DEV.tolist()),
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                    transform=transform)

    # Use Dirichlet Non-IID partitioning
    client_subsets = partition_dirichlet(
        dataset=train_dataset,
        num_clients=config.num_clients,
        alpha=config.dirichlet_alpha,
        seed=config.seed,
    )

    client_loaders = {}
    client_train_indices = {}
    for client_id, subset in enumerate(client_subsets):
        client_loaders[client_id] = DataLoader(
            subset, batch_size=config.batch_size, shuffle=True
        )
        client_train_indices[client_id] = [int(i) for i in subset.indices]

    # Test set split setup
    test_x = torch.stack([item[0] for item in test_dataset])
    test_y = torch.tensor([item[1] for item in test_dataset], dtype=torch.long)

    kd_indices, eval_indices = split_test_indices(
        test_y.numpy(), config.kd_reference_size, config.test_split_seed)

    eval_count = min(1000, len(eval_indices))
    diag = torch.tensor(eval_indices[:eval_count], dtype=torch.long)
    clean_test_loader = DataLoader(
        TensorDataset(test_x[diag], test_y[diag]),
        batch_size=1000, shuffle=False
    )

    data_info = {
        "train_partition": client_train_indices,
        "test_split": {
            "seed": config.test_split_seed,
            "kd_reference_size": config.kd_reference_size,
            "kd_indices": kd_indices,
            "eval_indices": eval_indices,
        },
    }
    return client_loaders, clean_test_loader, data_info


def make_backdoor_test_loader(test_dataset, trigger, indices, batch_size=1000,
                              max_samples=1000):
    """Triggered held-out images labelled with the attack target.

    Images whose true class already equals the target are excluded: a model
    that predicts the target for them is simply correct, not backdoored, so
    keeping them would inflate the attack success rate.
    """
    targets = test_dataset.targets
    chosen = [i for i in indices if int(targets[i]) != int(trigger.target_label)]
    chosen = chosen[:max_samples]
    data = torch.stack([test_dataset[i][0] for i in chosen])
    labels = torch.full((len(chosen),), trigger.target_label, dtype=torch.long)
    poisoned = torch.stack([apply_trigger(image, trigger) for image in data])
    return DataLoader(TensorDataset(poisoned, labels), batch_size=batch_size, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comm_round", "--rounds", dest="rounds", type=int, default=10)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=2500)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--pmr", type=float, default=0.05)
    parser.add_argument("--poison_rate", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Dirichlet concentration parameter for non-IID split")
    parser.add_argument(
    "--partition",type=str,choices=["iid", "dirichlet"],default="iid",help="Dataset partitioning strategy: iid or dirichlet")
    parser.add_argument("--attack_start_round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--optimizer_type", type=str, default="SGD")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--final_m_rule", type=str, default="frequency",
                        choices=["union", "frequency"])
    parser.add_argument("--final_m_tau", type=float, default=0.5)
    parser.add_argument("--kd_reference_size", type=int, default=2500)
    parser.add_argument("--test_split_seed", type=int, default=0)
    parser.add_argument("--tamper_test", action="store_true",
                        help="commit-reveal self-test; do NOT use for real results")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        num_clients=args.num_clients,
        samples_per_client=args.samples_per_client,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rounds=args.rounds,
        pmr=args.pmr,
        poison_rate=args.poison_rate,
        alpha=args.alpha,
        dirichlet_alpha=args.dirichlet_alpha,  # <--- Pass it directly here
        partition=args.partition,
        attack_start_round=args.attack_start_round,
        seed=args.seed,
        output_dir=args.output_dir,
        final_m_rule=args.final_m_rule,
        final_m_tau=args.final_m_tau,
        kd_reference_size=args.kd_reference_size,
        test_split_seed=args.test_split_seed,
        tamper_test=args.tamper_test,
    )
    config.validate()
    if config.tamper_test:
        print("!" * 60)
        print("TAMPER TEST ENABLED: one reveal per round is corrupted on purpose.")
        print("Results from this run must NOT be used.")
        print("!" * 60)
    seed_random_generators(config.seed)

    aggregator_object = Aggregator()
    aggregator_object.private_attributes = {}
    collaborator_names = [
        f"benign_{i:02d}" for i in range(config.num_benign_clients)
    ] + [
        f"malicious_{i:02d}" for i in range(config.num_malicious_clients)
    ]
    collaborators = [Collaborator(name=name) for name in collaborator_names]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, clean_test_loader, data_info = build_datasetsIID(config) if partition == "iid" else build_datasetsDirichlet(config)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.tolist(), STD_DEV.tolist()),
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                    transform=transform)
    trigger = sample_trigger(config.trigger_size, config.target_label, seed=config.seed)

    backdoor_loader = make_backdoor_test_loader(
        test_dataset, trigger, data_info["test_split"]["eval_indices"])
    for idx, collab in enumerate(collaborators):
        train_loader = client_loaders[idx]
        collab.private_attributes = {
            "train_loader": train_loader,
            "test_loader": clean_test_loader,
            "backdoor_test_loader": backdoor_loader,
            "vote_vault": {},      # per-collaborator secret votes between commit and reveal
        }

    local_runtime = LocalRuntime(
        aggregator=aggregator_object,
        collaborators=collaborators,
    )

    model = LightweightResNet18()
    optimizer_templates = {
        collaborator.name: args.optimizer_type.upper()
        for collaborator in collaborators
    }

    run_info = {
        "client_train_indices": {
            collaborator_names[i]: data_info["train_partition"][i]
            for i in range(config.num_clients)
        },
        "test_split": data_info["test_split"],
    }
    flflow = FederatedFlow(
        model=model,
        optimizer_templates=optimizer_templates,
        config=config,
        device=device,
        run_info=run_info,
    )
    flflow.runtime = local_runtime
    flflow.run()


if __name__ == "__main__":
    main()
