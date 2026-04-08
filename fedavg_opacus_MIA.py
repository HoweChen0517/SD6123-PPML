import argparse
import os
import time

import torch
import torch.nn as nn

from models.cnn import SimpleCNN
from utils.data_utils import describe_partition, prepare_datasets
from utils.json_utils import generate_json_config
from utils.mia_utils import (
    calculate_fedavg_weights,
    collect_fixed_samples,
    compute_score_stats,
    compute_train_loss,
    evaluate_mia_round,
    fedavg,
    summarize_privacy,
)
from utils.opacus_client import OpacusClient
from utils.training import (
    EarlyStopper,
    evaluate_model,
    model_num_bytes,
    resolve_device,
    sample_client_indices,
    set_seed,
    write_jsonl,
)


def run(args):
    print(f"Using device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    set_seed(args.seed)
    bundle = prepare_datasets(args)
    model_fn = lambda: SimpleCNN(bundle.input_channels, bundle.num_classes, bundle.input_size)
    global_model = model_fn().to(args.device)
    clients = [
        OpacusClient(
            args,
            idx,
            model_fn,
            bundle.client_loaders[idx],
            bundle.client_val_loaders[idx],
            bundle.client_test_loaders[idx],
        )
        for idx in range(args.num_clients)
    ]

    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
    for client in clients:
        client.set_parameters(global_state)

    criterion = nn.CrossEntropyLoss()
    result_path = os.path.join(
        "result",
        "fedavg_opacus_MIA",
        f"{args.dataset}_{args.partition}_sigma({args.dp_noise_multiplier})_q({args.dp_sample_rate})_C({args.dp_max_grad_norm}).jsonl",
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    write_jsonl(
        result_path,
        {
            **generate_json_config(args),
            "partition_summary": describe_partition(bundle.client_loaders),
            "model_bytes": model_num_bytes(global_model),
            "dp_backend": "opacus",
            "dp_accountant": args.dp_accountant,
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

    stopper = EarlyStopper(args.patience, args.early_stop_burn_in)
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
            pos_stats, neg_stats = compute_score_stats(scores, labels)
            train_loss = compute_train_loss(target_client)
            val_loss = target_client.cal_val_loss()
            gap = val_loss - train_loss
            dp_epsilon = target_client.current_privacy_epsilon()
            print(
                f"[MIA] Round {round_idx}: AUC = {auc:.4f}, "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, epsilon={dp_epsilon:.4f}"
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
    parser = argparse.ArgumentParser(description="FedAvg with Opacus DP-SGD and Membership Inference Attack")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--partition", type=str, default="auto", choices=["auto", "iid", "shard"])
    parser.add_argument("--model", type=str, default="SimpleCNN")
    parser.add_argument("--optimizer", type=str, default="dp_sgd", choices=["dp_sgd"])
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
    parser.add_argument("--early_stop_burn_in", type=int, default=0)
    parser.add_argument("--dp_noise_multiplier", type=float, default=0.0)
    parser.add_argument("--dp_sample_rate", type=float, default=1.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--dp_max_grad_norm", type=float, default=1e6)
    parser.add_argument("--dp_accountant", type=str, default="rdp", choices=["rdp", "prv", "gdp"])
    parser.add_argument("--dp_secure_mode", action="store_true")
    parser.add_argument("--dp_grad_sample_mode", type=str, default="hooks", choices=["hooks", "ew"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.device = resolve_device(args.device, args.device_id)
    os.makedirs(os.path.join("result", "fedavg_opacus_MIA"), exist_ok=True)
    run(args)
