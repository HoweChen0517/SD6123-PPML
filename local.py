import argparse
import copy
import os
import time

import torch

from models.cnn import SimpleCNN
from utils.data_utils import describe_partition, prepare_datasets
from utils.json_utils import generate_json_config
from utils.training import EarlyStopper, evaluate_model, resolve_device, set_seed, write_jsonl


def run(args):
    set_seed(args.seed)
    bundle = prepare_datasets(args)
    model = SimpleCNN(bundle.input_channels, bundle.num_classes, bundle.input_size).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    result_path = os.path.join("result", "local", f"{args.dataset}_{args.partition}.jsonl")
    write_jsonl(
        result_path,
        {
            **generate_json_config(args),
            "partition_summary": describe_partition(bundle.client_loaders),
            "model_bytes": sum(param.numel() * param.element_size() for param in model.parameters()),
        },
        reset=True,
    )

    stopper = EarlyStopper(args.patience, args.early_stop_burn_in)
    total_train_time = 0.0
    for round_idx in range(args.global_rounds):
        start_time = time.time()
        model.train()
        for _ in range(args.local_epochs):
            for inputs, labels in bundle.centralized_train_loader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        total_train_time += time.time() - start_time

        val_loss, val_acc = evaluate_model(model, bundle.centralized_val_loader, args.device, criterion)
        test_loss, test_acc = evaluate_model(model, bundle.test_loader, args.device, criterion)
        improved, early_stop = stopper.step(val_loss)

        write_jsonl(
            result_path,
            {
                "round": round_idx,
                "train_time": total_train_time,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "improved": improved,
            },
        )

        print(
            f"Round {round_idx}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, "
            f"val_loss={val_loss:.4f}, best_val_loss={stopper.best_loss:.4f}"
        )

        if early_stop:
            print(f"Early stopping at round {round_idx}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OurProject centralized baseline")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--partition", type=str, default="auto", choices=["auto", "iid", "shard"])
    parser.add_argument("--model", type=str, default="SimpleCNN")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--global_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--shards_per_client", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stop_burn_in", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    args.device = resolve_device(args.device, args.device_id)
    os.makedirs(os.path.join("result", "local"), exist_ok=True)
    run(args)
