import argparse
import os
import time

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


def run(args):
    set_seed(args.seed)
    bundle = prepare_datasets(args)
    model_fn = lambda: SimpleCNN(bundle.input_channels, bundle.num_classes, bundle.input_size)
    global_model = model_fn().to(args.device)
    clients = [
        ClientProx(
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

    criterion = __import__("torch").nn.CrossEntropyLoss()
    result_path = os.path.join("result", "fedprox", f"{args.dataset}_{args.partition}.jsonl")
    write_jsonl(
        result_path,
        {
            **generate_json_config(args),
            "partition_summary": describe_partition(bundle.client_loaders),
            "model_bytes": model_num_bytes(global_model),
        },
        reset=True,
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

        weights = calculate_fedavg_weights(selected_clients)
        global_state = fedavg(weights, selected_clients, global_model, args.device)
        total_comm_cost += model_num_bytes(global_model) * 2 * len(selected_clients)
        for client in clients:
            client.set_parameters(global_state)

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
            f"client_mean_acc={sum(client_acc)/len(client_acc):.4f}, val_loss={val_loss:.4f}"
        )

        if early_stop:
            print(f"Early stopping at round {round_idx}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OurProject FedProx")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--partition", type=str, default="auto", choices=["auto", "iid", "shard"])
    parser.add_argument("--model", type=str, default="SimpleCNN")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--global_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--shards_per_client", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stop_burn_in", type=int, default=100)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.device = resolve_device(args.device, args.device_id)
    os.makedirs(os.path.join("result", "fedprox"), exist_ok=True)
    run(args)
