import math

import torch
from torch import nn
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from .training import create_optimizer, create_scheduler


def resolve_opacus_sample_rate(dp_sample_rate, batch_size, dataset_size):
    if dp_sample_rate is None:
        return min(1.0, batch_size / max(dataset_size, 1))
    return float(min(1.0, max(dp_sample_rate, 1e-8)))


def build_poisson_train_loader(train_loader, sample_rate):
    dataset = train_loader.dataset
    sampler = UniformWithReplacementSampler(
        num_samples=len(dataset),
        sample_rate=sample_rate,
        steps=max(1, math.ceil(1.0 / sample_rate)),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_loader.collate_fn,
        persistent_workers=getattr(train_loader, "persistent_workers", False),
    )


def disable_inplace_modules(module):
    for name, child in module.named_children():
        disable_inplace_modules(child)
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
    return module


def make_private_components(args, model, train_loader, local_epochs):
    model = disable_inplace_modules(model)
    model = ModuleValidator.fix(model).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    sample_rate = resolve_opacus_sample_rate(
        getattr(args, "dp_sample_rate", None), args.batch_size, len(train_loader.dataset)
    )
    private_loader = build_poisson_train_loader(train_loader, sample_rate)
    base_optimizer = create_optimizer(args, model.parameters())
    privacy_engine = PrivacyEngine(
        accountant=getattr(args, "dp_accountant", "rdp"),
        secure_mode=getattr(args, "dp_secure_mode", False),
    )
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=base_optimizer,
        criterion=criterion,
        data_loader=private_loader,
        noise_multiplier=args.dp_noise_multiplier,
        max_grad_norm=args.dp_max_grad_norm,
        poisson_sampling=False,
        clipping="flat",
        loss_reduction="mean",
        grad_sample_mode=getattr(args, "dp_grad_sample_mode", "hooks"),
    )
    total_steps = max(1, len(private_loader) * local_epochs * args.global_rounds)
    scheduler = create_scheduler(optimizer, total_steps)
    return model, optimizer, criterion, private_loader, scheduler, privacy_engine, sample_rate


def unwrap_model(model):
    return model._module if hasattr(model, "_module") else model


def clone_model_state(model):
    base_model = unwrap_model(model)
    return {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}


def load_model_state(model, state_dict, device):
    base_model = unwrap_model(model)
    base_model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})


def clone_trainable_parameters(model):
    base_model = unwrap_model(model)
    return [param.detach().clone() for param in base_model.parameters()]


def apply_proximal_step(model, reference_params, mu, optimizer):
    if mu <= 0:
        return
    lr = optimizer.param_groups[0]["lr"]
    base_model = unwrap_model(model)
    with torch.no_grad():
        for param, reference in zip(base_model.parameters(), reference_params):
            param.add_(reference.to(param.device) - param, alpha=lr * mu)


def safe_get_epsilon(privacy_engine, delta, steps):
    if privacy_engine is None or steps <= 0:
        return 0.0
    try:
        return float(privacy_engine.get_epsilon(delta))
    except (ValueError, ZeroDivisionError, OverflowError, RuntimeError):
        return float("inf")
