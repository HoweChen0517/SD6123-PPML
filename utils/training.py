import copy
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device, device_id):
    if device == "cuda":
        return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def clone_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def load_state_dict(model, state_dict, device):
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


def average_state_dicts(state_dicts, weights):
    avg_state = copy.deepcopy(state_dicts[0])
    for name in avg_state:
        avg_state[name] = torch.zeros_like(avg_state[name])
        for state, weight in zip(state_dicts, weights):
            avg_state[name] += state[name] * weight
    return avg_state


def model_num_bytes(model):
    return sum(param.numel() * param.element_size() for param in model.parameters())


def sample_client_indices(num_clients, clients_per_round, seed, round_idx):
    rng = random.Random(seed + round_idx)
    if clients_per_round >= num_clients:
        return list(range(num_clients))
    return sorted(rng.sample(range(num_clients), clients_per_round))


def create_optimizer(args, params):
    optim_name = args.optimizer.lower()
    if optim_name in {"sgd", "dp_sgd"}:
        return torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if optim_name == "adam":
        return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def create_scheduler(optimizer, total_steps):
    if total_steps <= 0:
        return None

    def lr_lambda(step):
        progress = step / total_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return avg_loss, accuracy


def write_jsonl(path, payload, reset=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if reset else "a"
    with open(path, mode, encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        should_stop = self.counter >= self.patience
        return improved, should_stop
