import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from .training import average_state_dicts


def calculate_fedavg_weights(clients):
    total = sum(client.num_train_samples for client in clients)
    return [client.num_train_samples / total for client in clients]


def fedavg(weights, client_objs, global_model, device):
    states = [client.get_parameters() for client in client_objs]
    global_state = average_state_dicts(states, weights)
    global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    for client in client_objs:
        client.set_parameters(global_state)
    return global_state


def collect_fixed_samples(client_loader, num_samples):
    samples = []
    samples_per_batch = 2
    for x, y in client_loader:
        batch_size = x.size(0)
        for j in range(min(samples_per_batch, batch_size)):
            samples.append((x[j : j + 1], y[j : j + 1]))
            if len(samples) >= num_samples:
                return samples[:num_samples]
    return samples


def compute_train_loss(client):
    client.model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, y in client.train_loader:
            x = x.to(client.device)
            y = y.to(client.device)
            logits = client.model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
    client.model.train()
    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate_mia_round(global_model, target_client, pos_samples, neg_samples, device, criterion):
    target_client.model.eval()
    global_model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for x, y in pos_samples:
            x = x.to(device)
            y = y.to(device)
            loss_global = criterion(global_model(x), y).item()
            loss_local = criterion(target_client.model(x), y).item()
            scores.append((loss_global - loss_local) / (loss_global + 1e-8))
            labels.append(1)

        for x, y in neg_samples:
            x = x.to(device)
            y = y.to(device)
            loss_global = criterion(global_model(x), y).item()
            loss_local = criterion(target_client.model(x), y).item()
            scores.append((loss_global - loss_local) / (loss_global + 1e-8))
            labels.append(0)

    auc = roc_auc_score(labels, scores)
    unique_scores = np.sort(np.unique(scores))
    candidate_thresholds = []
    if len(unique_scores) > 1:
        for idx in range(len(unique_scores) - 1):
            candidate_thresholds.append((unique_scores[idx] + unique_scores[idx + 1]) / 2)
    candidate_thresholds.append(unique_scores[0] - 1e-8)
    candidate_thresholds.append(unique_scores[-1] + 1e-8)

    best_acc = 0.0
    for threshold in candidate_thresholds:
        pred = [1 if score > threshold else 0 for score in scores]
        best_acc = max(best_acc, accuracy_score(labels, pred))

    target_client.model.train()
    global_model.train()
    return auc, best_acc, scores, labels


def compute_score_stats(scores, labels):
    def summarize(values):
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            return {}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "q3": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
        }

    pos_scores = [score for score, label in zip(scores, labels) if label == 1]
    neg_scores = [score for score, label in zip(scores, labels) if label == 0]
    return summarize(pos_scores), summarize(neg_scores)


def summarize_privacy(clients, selected_ids):
    epsilons = [clients[idx].current_privacy_epsilon() for idx in selected_ids]
    epsilons = [eps for eps in epsilons if eps is not None]
    if not epsilons:
        return {}
    return {
        "dp_epsilon_selected_mean": float(np.mean(epsilons)),
        "dp_epsilon_selected_max": float(np.max(epsilons)),
    }


def summarize_ditto_privacy(clients, selected_ids):
    global_eps = [clients[idx].current_privacy_epsilon() for idx in selected_ids]
    global_eps = [eps for eps in global_eps if eps is not None]
    personal_eps = [clients[idx].current_personal_privacy_epsilon() for idx in selected_ids]
    personal_eps = [eps for eps in personal_eps if eps is not None]

    stats = {}
    if global_eps:
        stats["dp_epsilon_selected_mean"] = float(np.mean(global_eps))
        stats["dp_epsilon_selected_max"] = float(np.max(global_eps))
    if personal_eps:
        stats["dp_personal_epsilon_selected_mean"] = float(np.mean(personal_eps))
        stats["dp_personal_epsilon_selected_max"] = float(np.max(personal_eps))
    return stats
