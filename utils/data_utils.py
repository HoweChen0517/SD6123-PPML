import io
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required to decode the local parquet image datasets.") from exc


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    test_loader: DataLoader
    centralized_train_loader: DataLoader
    centralized_val_loader: DataLoader
    client_loaders: list
    client_val_loaders: list
    client_test_loaders: list
    client_train_sizes: list
    num_classes: int
    input_channels: int
    input_size: int
    class_names: list


class LocalParquetImageDataset(Dataset):
    def __init__(self, rows, image_key, transform):
        self.rows = rows
        self.image_key = image_key
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = decode_image(row[self.image_key])
        label = int(row["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def decode_image(value):
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])

    if hasattr(value, "as_py"):
        return decode_image(value.as_py())

    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value))

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return Image.fromarray(value.astype(np.uint8), mode="L")
        return Image.fromarray(value.astype(np.uint8))

    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


def load_parquet_rows(path):
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        return table.to_pylist()
    except ImportError:
        pass

    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    except ImportError as exc:
        raise ImportError(
            "Loading the local parquet datasets requires either `pyarrow` or `pandas`."
        ) from exc


def build_transforms(dataset_name):
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torchvision is required for image preprocessing.") from exc

    if dataset_name == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return train_transform, test_transform, 1, 28

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_transform, test_transform, 3, 32


def resolve_partition(dataset_name, partition):
    if partition != "auto":
        return partition
    return "iid" if dataset_name == "mnist" else "shard"


def load_dataset(dataset_name):
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), dataset_name, "data")
    train_path = os.path.join(root, "train-00000-of-00001.parquet")
    test_path = os.path.join(root, "test-00000-of-00001.parquet")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files not found under {root}")

    train_rows = load_parquet_rows(train_path)
    test_rows = load_parquet_rows(test_path)

    image_key = "image" if dataset_name == "mnist" else "img"
    class_names = [str(i) for i in range(10)] if dataset_name == "mnist" else [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    return train_rows, test_rows, image_key, class_names


def iid_partition(labels, num_clients, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]


def shard_partition(labels, num_clients, shards_per_client, seed):
    num_shards = num_clients * shards_per_client
    indices = np.arange(len(labels))
    sorted_indices = indices[np.argsort(labels)]
    shards = np.array_split(sorted_indices, num_shards)
    rng = np.random.default_rng(seed)
    shard_order = np.arange(num_shards)
    rng.shuffle(shard_order)

    client_indices = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        chosen = shard_order[client_id * shards_per_client : (client_id + 1) * shards_per_client]
        merged = np.concatenate([shards[idx] for idx in chosen]).astype(int)
        rng.shuffle(merged)
        client_indices[client_id] = merged.tolist()
    return client_indices


def split_train_val(indices, val_ratio, seed):
    indices = list(indices)
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    if len(indices) <= 1:
        return indices, indices
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices = val_indices
    return train_indices, val_indices


def build_loader(dataset, indices, batch_size, shuffle, num_workers):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_datasets(args):
    dataset_name = args.dataset.lower()
    partition = resolve_partition(dataset_name, args.partition)
    args.partition = partition

    train_rows, test_rows, image_key, class_names = load_dataset(dataset_name)
    train_transform, test_transform, input_channels, input_size = build_transforms(dataset_name)

    train_dataset = LocalParquetImageDataset(train_rows, image_key=image_key, transform=train_transform)
    test_dataset = LocalParquetImageDataset(test_rows, image_key=image_key, transform=test_transform)

    labels = np.array([int(row["label"]) for row in train_rows])
    if partition == "iid":
        client_indices = iid_partition(labels, args.num_clients, args.seed)
    else:
        client_indices = shard_partition(labels, args.num_clients, args.shards_per_client, args.seed)

    client_loaders = []
    client_val_loaders = []
    client_test_loaders = []
    client_train_sizes = []

    test_indices = np.arange(len(test_rows))
    test_splits = np.array_split(test_indices, args.num_clients)

    for client_id, indices in enumerate(client_indices):
        train_indices, val_indices = split_train_val(indices, args.val_ratio, args.seed + client_id)
        client_loaders.append(build_loader(train_dataset, train_indices, args.batch_size, True, args.num_workers))
        client_val_loaders.append(build_loader(train_dataset, val_indices, args.batch_size, False, args.num_workers))
        client_test_loaders.append(build_loader(test_dataset, test_splits[client_id], args.batch_size, False, args.num_workers))
        client_train_sizes.append(len(train_indices))

    full_train_indices = np.arange(len(train_rows))
    centralized_train_indices, centralized_val_indices = split_train_val(full_train_indices, args.val_ratio, args.seed)

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        centralized_train_loader=build_loader(
            train_dataset, centralized_train_indices, args.batch_size, True, args.num_workers
        ),
        centralized_val_loader=build_loader(
            train_dataset, centralized_val_indices, args.batch_size, False, args.num_workers
        ),
        client_loaders=client_loaders,
        client_val_loaders=client_val_loaders,
        client_test_loaders=client_test_loaders,
        client_train_sizes=client_train_sizes,
        num_classes=len(class_names),
        input_channels=input_channels,
        input_size=input_size,
        class_names=class_names,
    )


def describe_partition(client_loaders):
    sizes = [len(loader.dataset) for loader in client_loaders]
    return {
        "num_clients": len(sizes),
        "min_train_size": min(sizes),
        "max_train_size": max(sizes),
        "mean_train_size": float(sum(sizes) / max(len(sizes), 1)),
        "std_train_size": float(np.std(sizes)) if sizes else 0.0,
    }
