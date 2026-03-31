def generate_json_config(args):
    return {
        "model": args.model,
        "dataset": args.dataset,
        "partition": args.partition,
        "num_clients": args.num_clients,
        "clients_per_round": args.clients_per_round,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "local_epochs": args.local_epochs,
        "global_rounds": args.global_rounds,
        "seed": args.seed,
        "mu": getattr(args, "mu", None),
        "plocal_epochs": getattr(args, "plocal_epochs", None),
    }
