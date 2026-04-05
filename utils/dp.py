import math

import torch


def is_dp_optimizer(optimizer_name):
    return optimizer_name.lower() == "dp_sgd"


def resolve_dp_sample_rate(dp_sample_rate, batch_size, dataset_size):
    if dp_sample_rate is None:
        return min(1.0, batch_size / max(dataset_size, 1))
    return min(max(float(dp_sample_rate), 0.0), 1.0)


def compute_dp_epsilon(sample_rate, noise_multiplier, steps, delta):
    """A coarse upper bound for subsampled Gaussian DP-SGD.

    This is a lightweight approximation intended for experiment tracking,
    not a formal accountant.
    """
    if steps <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return float("inf")
    if not 0 < delta < 1:
        return float("inf")

    sample_rate = min(max(sample_rate, 0.0), 1.0)
    leading = sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier
    correction = steps * (sample_rate ** 2) / max(noise_multiplier ** 2, 1e-12)
    return leading + correction


def _clip_and_accumulate(trainable_params, aggregated_grads, max_grad_norm):
    total_norm_sq = 0.0
    sample_grads = []
    for param in trainable_params:
        if param.grad is None:
            grad = torch.zeros_like(param)
        else:
            grad = param.grad.detach().clone()
        sample_grads.append(grad)
        total_norm_sq += grad.pow(2).sum().item()

    total_norm = math.sqrt(total_norm_sq)
    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-12))
    for aggregated, grad in zip(aggregated_grads, sample_grads):
        aggregated.add_(grad, alpha=clip_coef)


def dp_sgd_step(
    model,
    optimizer,
    criterion,
    inputs,
    labels,
    max_grad_norm,
    noise_multiplier,
    sample_rate,
    scheduler=None,
    loss_builder=None,
):
    model.train()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        return 0

    if sample_rate < 1.0:
        keep_mask = torch.rand(inputs.size(0), device=inputs.device) < sample_rate
        if not keep_mask.any():
            keep_mask[torch.randint(0, inputs.size(0), (1,), device=inputs.device)] = True
        inputs = inputs[keep_mask]
        labels = labels[keep_mask]

    selected_count = labels.size(0)
    aggregated_grads = [torch.zeros_like(param) for param in trainable_params]

    for sample_input, sample_label in zip(inputs, labels):
        optimizer.zero_grad(set_to_none=True)
        sample_input = sample_input.unsqueeze(0)
        sample_label = sample_label.unsqueeze(0)
        if loss_builder is None:
            logits = model(sample_input)
            loss = criterion(logits, sample_label)
        else:
            loss = loss_builder(model, sample_input, sample_label, selected_count)
        loss.backward()
        _clip_and_accumulate(trainable_params, aggregated_grads, max_grad_norm)

    optimizer.zero_grad(set_to_none=True)
    noise_std = noise_multiplier * max_grad_norm
    for param, aggregated in zip(trainable_params, aggregated_grads):
        if noise_std > 0:
            aggregated = aggregated + torch.randn_like(aggregated) * noise_std
        param.grad = aggregated / max(selected_count, 1)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return selected_count
