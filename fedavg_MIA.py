import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from models.cnn import SimpleCNN
from utils.client import Client
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
    samples_per_batch = 2  # 每个 batch 取几个样本
    for i, (x, y) in enumerate(client_loader):
        batch_size = x.size(0)
        # 取前 samples_per_batch 个（或随机取）
        for j in range(min(samples_per_batch, batch_size)):
            samples.append((x[j:j+1], y[j:j+1]))
            if len(samples) >= num_samples:
                return samples[:num_samples]
    return samples


def compute_train_loss(client):
    """计算客户端在当前模型下的平均训练损失（遍历整个 train_loader）"""
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
    """
    方案A：使用全局模型作为基准，评估当前轮次的成员推理攻击性能。

    Args:
        global_model: 上一轮聚合后的全局模型（w_global）
        target_client: 被攻击的客户端对象（已经完成本地训练）
        pos_samples: 正例样本列表，属于 target_client 的训练集
        neg_samples: 反例样本列表，不属于 target_client 的训练集（例如来自 client_1）
        device: 计算设备
        criterion: 损失函数

    Returns:
        auc: ROC AUC 分数
        best_acc: 最佳阈值下的准确率
        scores: 每个样本的攻击得分列表
        labels: 每个样本的真实标签（1=成员，0=非成员）
    """
    target_client.model.eval()
    global_model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        # 处理正例（成员）
        for x, y in pos_samples:
            x, y = x.to(device), y.to(device)
            loss_global = criterion(global_model(x), y).item()
            loss_local = criterion(target_client.model(x), y).item()
            # 相对下降率，分母加小量避免除零
            score = (loss_global - loss_local) / (loss_global + 1e-8)
            scores.append(score)
            labels.append(1)

        # 处理反例（非成员）
        for x, y in neg_samples:
            x, y = x.to(device), y.to(device)
            loss_global = criterion(global_model(x), y).item()
            loss_local = criterion(target_client.model(x), y).item()
            score = (loss_global - loss_local) / (loss_global + 1e-8)
            scores.append(score)
            labels.append(0)

    # 计算 AUC
    auc = roc_auc_score(labels, scores)

    # 搜索最佳阈值下的准确率
    best_acc = 0.0
    thresholds = np.linspace(min(scores), max(scores), 100)
    for thresh in thresholds:
        pred = [1 if s > thresh else 0 for s in scores]
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc = acc

    # 恢复训练模式
    target_client.model.train()
    global_model.train()

    return auc, best_acc, scores, labels


def run(args):
    print(f"Using device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    set_seed(args.seed)
    bundle = prepare_datasets(args)
    model_fn = lambda: SimpleCNN(bundle.input_channels, bundle.num_classes, bundle.input_size)
    global_model = model_fn().to(args.device)
    clients = [
        Client(args, idx, model_fn, bundle.client_loaders[idx], bundle.client_val_loaders[idx],
               bundle.client_test_loaders[idx])
        for idx in range(args.num_clients)
    ]

    # 初始化全局参数
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    for client in clients:
        client.set_parameters(global_state)

    criterion = nn.CrossEntropyLoss()
    result_path = os.path.join("result", "fedavg_MIA", f"{args.dataset}_{args.partition}.jsonl")
    write_jsonl(
        result_path,
        {
            **generate_json_config(args),
            "partition_summary": describe_partition(bundle.client_loaders),
            "model_bytes": model_num_bytes(global_model),
        },
        reset=True,
    )

    # ---------- MIA 准备：固定正负样本 ----------
    attack_client_id = 0  # 被攻击的客户端
    reference_client_id = 1  # 用于提供反例的客户端（确保其训练样本不在 attack_client 中）
    num_mia_samples = 200  # 正例和反例各取多少个样本

    # 从 attack_client 的训练集中取正例
    pos_samples = collect_fixed_samples(bundle.client_loaders[attack_client_id], num_mia_samples)
    # 从 reference_client 的训练集中取反例
    neg_samples = collect_fixed_samples(bundle.client_loaders[reference_client_id], num_mia_samples)

    print(f"[MIA] 已收集正例 {len(pos_samples)} 个（来自 client_{attack_client_id}），"
          f"反例 {len(neg_samples)} 个（来自 client_{reference_client_id}）。")
    # -----------------------------------------



    stopper = EarlyStopper(args.patience)
    total_train_time = 0.0
    total_comm_cost = 0

    for round_idx in range(args.global_rounds):
        selected_ids = sample_client_indices(args.num_clients, args.clients_per_round, args.seed, round_idx)
        selected_clients = [clients[idx] for idx in selected_ids]

        # ---------- 本地训练 ----------
        start_time = time.time()
        for client in selected_clients:
            client.fine_tune()
        total_train_time += time.time() - start_time

        # ---------- 成员推理攻击（仅在 attack_client 被选中时执行）----------
        if attack_client_id in selected_ids:
            # 此时 global_model 仍然是上一轮聚合后的模型（w_global）
            # target_client 是已经完成本地训练的 client_0
            target_client = clients[attack_client_id]
            auc, best_acc, scores, labels = evaluate_mia_round(
                global_model, target_client, pos_samples, neg_samples,
                args.device, criterion
            )
            print(f"[MIA] Round {round_idx}: AUC = {auc:.4f}")

            # 分离正例和反例的得分，并保留两位小数
            pos_scores = [round(score, 2) for score, label in zip(scores, labels) if label == 1]
            neg_scores = [round(score, 2) for score, label in zip(scores, labels) if label == 0]

            def print_stats(name, data):
                if not data:
                    print(f"{name}: 无数据")
                    return
                arr = np.array(data)
                mean = np.mean(arr)
                std = np.std(arr, ddof=1)  # 样本标准差
                min_val = np.min(arr)
                max_val = np.max(arr)
                q1, median, q3 = np.percentile(arr, [25, 50, 75])
                print(f"{name} – Mean: {mean:.4f}, Std: {std:.4f}")
                print(f"      Min: {min_val:.4f}, Q1: {q1:.4f}, Median: {median:.4f}, Q3: {q3:.4f}, Max: {max_val:.4f}")

            print_stats("Pos Scores", pos_scores)
            print_stats("Neg Scores", neg_scores)
            # 计算训练损失和验证损失（对 target_client）
            train_loss = compute_train_loss(target_client)
            val_loss = target_client.cal_val_loss()   # ClientDitto 应支持该方法
            gap = val_loss - train_loss
            print(f"[Client0] Round {round_idx}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, gap={gap:.4f}")


            # 计算训练损失和验证损失
            train_loss = compute_train_loss(target_client)
            val_loss = target_client.cal_val_loss()  # 已有方法
            gap = val_loss - train_loss
            print(f"[Client0] Round {round_idx}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, gap={gap:.4f}")

            # 将攻击结果也写入 jsonl 文件（可选）
            write_jsonl(
                result_path,
                {
                    "mia_round": round_idx,
                    "auc": auc,
                    "mia_best_accuracy": best_acc,
                    "attack_client": attack_client_id,
                    "num_pos_samples": len(pos_samples),
                    "num_neg_samples": len(neg_samples),
                    "train_loss_client0": train_loss,  # 新增
                    "val_loss_client0": val_loss,  # 新增
                    "overfit_gap": gap,  # 新增
                },
                reset=False,
            )
        # ---------------------------------

        # ---------- FedAvg 聚合 ----------
        weights = calculate_fedavg_weights(selected_clients)
        global_state = fedavg(weights, selected_clients, global_model, args.device)
        total_comm_cost += model_num_bytes(global_model) * 2 * len(selected_clients)
        # 将新全局模型同步给所有客户端
        for client in clients:
            client.set_parameters(global_state)

        # ---------- 原有评估与日志 ----------
        val_loss = sum(client.cal_val_loss() for client in selected_clients) / max(len(selected_clients), 1)
        test_loss, test_acc = evaluate_model(global_model, bundle.test_loader, args.device, criterion)
        client_acc = [client.test()[1] for client in clients]
        improved, early_stop = stopper.step(val_loss)

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
            },
        )

        print(
            f"Round {round_idx}: global_test_acc={test_acc:.4f}, "
            f"client_mean_acc={sum(client_acc) / len(client_acc):.4f}, val_loss={val_loss:.4f}"
        )

        # if early_stop:
        #     print(f"Early stopping at round {round_idx}")
        #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedAvg with Membership Inference Attack (Scheme A)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--partition", type=str, default="auto", choices=["auto", "iid", "shard"])
    parser.add_argument("--model", type=str, default="SimpleCNN")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=5,  # 注意：为了攻击效果，建议设大一点（如5）
                        help="Local epochs, increase to make overfitting more obvious")
    parser.add_argument("--global_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--shards_per_client", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.device = resolve_device(args.device, args.device_id)
    os.makedirs(os.path.join("result", "fedavg"), exist_ok=True)
    run(args)