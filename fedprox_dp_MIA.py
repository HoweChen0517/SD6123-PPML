import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from models.cnn import SimpleCNN
from utils.clientprox import ClientProx
from utils.data_utils import describe_partition, prepare_datasets
from utils.json_utils import generate_json_config
from utils.training import (
    EarlyStopper,
    average_state_dicts,
    evaluate_model,
    model_num_bytes,
    resolve_device,
    sample_client_indices,
    set_seed,
    write_jsonl,
)


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
            samples.append((x[j:j + 1], y[j:j + 1]))
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
            x, y = x.to(client.device), y.to(client.device)
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
            x, y = x.to(device), y.to(device)
            loss_global = criterion(global_model(x), y).item()
            loss_local = criterion(target_client.model(x), y).item()
            scores.append((loss_global - loss_local) / (loss_global + 1e-8))
            labels.append(1)
        for x, y in neg_samples:
            x, y = x.to(device), y.to(device)
            loss_global = criterion(global_model(x), y).item()
            loss_local = criterion(target_client.model(x), y).item()
            scores.append((loss_global - loss_local) / (loss_global + 1e-8))
            labels.append(0)

    auc = roc_auc_score(labels, scores)
    unique_scores = np.sort(np.unique(scores))
    candidate_thresholds = []
    if len(unique_scores) > 1:
        for i in range(len(unique_scores) - 1):
            candidate_thresholds.append((unique_scores[i] + unique_scores[i + 1]) / 2)
    candidate_thresholds.append(unique_scores[0] - 1e-8)
    candidate_thresholds.append(unique_scores[-1] + 1e-8)

    best_acc = 0.0
    for thresh in candidate_thresholds:
        pred = [1 if s > thresh else 0 for s in scores]
        best_acc = max(best_acc, accuracy_score(labels, pred))

    target_client.model.train()
    global_model.train()
    return auc, best_acc, scores, labels


def summarize_privacy(clients, selected_ids):
    epsilons = [clients[idx].current_privacy_epsilon() for idx in selected_ids]
    epsilons = [eps for eps in epsilons if eps is not None]
    if not epsilons:
        return {}
    return {
        "dp_epsilon_selected_mean": float(np.mean(epsilons)),
        "dp_epsilon_selected_max": float(np.max(epsilons)),
    }


def run(args):
    print(f"Using device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    set_seed(args.seed)
    bundle = prepare_datasets(args)
    model_fn = lambda: SimpleCNN(bundle.input_channels, bundle.num_classes, bundle.input_size)
    global_model = model_fn().to(args.device)
    clients = [
        ClientProx(args, idx, model_fn, bundle.client_loaders[idx], bundle.client_val_loaders[idx], bundle.client_test_loaders[idx])
        for idx in range(args.num_clients)
    ]

    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    for client in clients:
        client.set_parameters(global_state)

    criterion = nn.CrossEntropyLoss()
    result_path = os.path.join("result", "fedprox_dp_MIA", f"{args.dataset}_{args.partition}.jsonl")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    write_jsonl(
        result_path,
        {
            **generate_json_config(args),
            "partition_summary": describe_partition(bundle.client_loaders),
            "model_bytes": model_num_bytes(global_model),
        },
        reset=True,
    )

    attack_client_id = 0
    reference_client_id = 1
    num_mia_samples = 200
    pos_samples = collect_fixed_samples(bundle.client_loaders[attack_client_id], num_mia_samples)
    neg_samples = collect_fixed_samples(bundle.client_loaders[reference_client_id], num_mia_samples)
    print(
        f"[MIA] 已收集正例 {len(pos_samples)} 个（来自 client_{attack_client_id}），"
        f"反例 {len(neg_samples)} 个（来自 client_{reference_client_id}）。"
    )

    stopper = EarlyStopper(args.patience)
    total_train_time = 0.0
    total_comm_cost = 0

    for round_idx in range(args.global_rounds):
        selected_ids = sample_client_indices(args.num_clients, args.clients_per_round, args.seed, round_idx)
        selected_clients = [clients[idx] for idx in selected_ids]

        start_time = time.time()
        for client in selected_clients:
            client.fine_tune()
        total_train_time += time.time() - start_time

        if attack_client_id in selected_ids:
            target_client = clients[attack_client_id]
            auc, best_acc, scores, labels = evaluate_mia_round(
                global_model, target_client, pos_samples, neg_samples, args.device, criterion
            )
            print(f"[MIA] Round {round_idx}: AUC = {auc:.4f}")

            pos_scores = [score for score, label in zip(scores, labels) if label == 1]
            neg_scores = [score for score, label in zip(scores, labels) if label == 0]

            def compute_stats(arr):
                arr = np.array(arr)
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "q1": float(np.percentile(arr, 25)),
                    "median": float(np.percentile(arr, 50)),
                    "q3": float(np.percentile(arr, 75)),
                    "max": float(np.max(arr)),
                }

            pos_stats = compute_stats(pos_scores)
            neg_stats = compute_stats(neg_scores)
            print(f"Pos Scores – Mean: {pos_stats['mean']:.4f}, Std: {pos_stats['std']:.4f}")
            print(
                f"      Min: {pos_stats['min']:.4f}, Q1: {pos_stats['q1']:.4f}, Median: {pos_stats['median']:.4f}, "
                f"Q3: {pos_stats['q3']:.4f}, Max: {pos_stats['max']:.4f}"
            )
            print(f"Neg Scores – Mean: {neg_stats['mean']:.4f}, Std: {neg_stats['std']:.4f}")
            print(
                f"      Min: {neg_stats['min']:.4f}, Q1: {neg_stats['q1']:.4f}, Median: {neg_stats['median']:.4f}, "
                f"Q3: {neg_stats['q3']:.4f}, Max: {neg_stats['max']:.4f}"
            )

            train_loss = compute_train_loss(target_client)
            val_loss = target_client.cal_val_loss()
            gap = val_loss - train_loss
            dp_epsilon = target_client.current_privacy_epsilon()
            print(
                f"[Client0] Round {round_idx}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, gap={gap:.4f}, dp_epsilon={dp_epsilon:.4f}"
            )

            write_jsonl(
                result_path,
                {
                    "mia_round": round_idx,
                    "auc": auc,
                    "mia_best_accuracy": best_acc,
                    "attack_client": attack_client_id,
                    "num_pos_samples": len(pos_samples),
                    "num_neg_samples": len(neg_samples),
                    "pos_scores_stats": pos_stats,
                    "neg_scores_stats": neg_stats,
                    "train_loss_client0": train_loss,
                    "val_loss_client0": val_loss,
                    "overfit_gap": gap,
                    "dp_epsilon_attack_client": dp_epsilon,
                },
                reset=False,
            )

        weights = calculate_fedavg_weights(selected_clients)
        global_state = fedavg(weights, selected_clients, global_model, args.device)
        total_comm_cost += model_num_bytes(global_model) * 2 * len(selected_clients)
        for client in clients:
            client.set_parameters(global_state)

        val_loss = sum(client.cal_val_loss() for client in selected_clients) / max(len(selected_clients), 1)
        test_loss, test_acc = evaluate_model(global_model, bundle.test_loader, args.device, criterion)
        client_acc = [client.test()[1] for client in clients]
        improved, early_stop = stopper.step(val_loss)
        privacy_stats = summarize_privacy(clients, selected_ids)

        write_jsonl(
            result_path,
            {
                "round": round_idx,
                "selected_clients": selected_ids,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "client_test_accuracy_mean": sum(client_acc) / len(client_acc),
                "client_test_accuracy": client_acc,
                "train_time": total_train_time,
                "communication_cost_bytes": total_comm_cost,
                "improved": improved,
                **privacy_stats,
            },
        )

        print(
            f"Round {round_idx}: global_test_acc={test_acc:.4f}, "
            f"client_mean_acc={sum(client_acc) / len(client_acc):.4f}, val_loss={val_loss:.4f}"
        )

        if early_stop:
            print(f"Early stopping at round {round_idx}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedProx with DP-SGD and Membership Inference Attack")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--partition", type=str, default="auto", choices=["auto", "iid", "shard"])
    parser.add_argument("--model", type=str, default="SimpleCNN")
    parser.add_argument("--optimizer", type=str, default="dp_sgd", choices=["sgd", "adam", "adamw", "dp_sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--global_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--shards_per_client", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.01, help="Proximal term coefficient for FedProx")
    parser.add_argument("--dp_noise_multiplier", type=float, default=1.0, help="Gaussian noise multiplier sigma.")
    parser.add_argument("--dp_sample_rate", type=float, default=None, help="Poisson subsampling rate q within each local batch.")
    parser.add_argument("--dp_delta", type=float, default=1e-5, help="Target delta for approximate privacy accounting.")
    parser.add_argument("--dp_max_grad_norm", type=float, default=1.0, help="Per-sample clipping norm C for DP-SGD.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.device = resolve_device(args.device, args.device_id)
    os.makedirs(os.path.join("result", "fedprox_dp_MIA"), exist_ok=True)
    run(args)
