import copy

import torch
from torch import nn

from .dp import compute_dp_epsilon, dp_sgd_step, is_dp_optimizer, resolve_dp_sample_rate
from .training import create_optimizer, create_scheduler, evaluate_model


class Client(nn.Module):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__()
        self.args = args
        self.id = client_id
        self.device = args.device
        self.local_epochs = args.local_epochs
        self.grad_clip = args.grad_clip
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model_fn().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.use_dp = is_dp_optimizer(args.optimizer)
        self.dp_noise_multiplier = getattr(args, "dp_noise_multiplier", 0.0)
        self.dp_max_grad_norm = getattr(args, "dp_max_grad_norm", self.grad_clip)
        self.dp_sample_rate = resolve_dp_sample_rate(
            getattr(args, "dp_sample_rate", None), args.batch_size, len(self.train_loader.dataset)
        )
        default_delta = 1.0 / max(len(self.train_loader.dataset), 1)
        self.dp_delta = getattr(args, "dp_delta", default_delta)
        self.dp_steps = 0
        self.optimizer = create_optimizer(args, self.model.parameters())
        total_steps = max(len(self.train_loader) * self.local_epochs * args.global_rounds, 1)
        self.scheduler = create_scheduler(self.optimizer, total_steps)

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def set_parameters(self, state_dict):
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()})

    def get_parameters(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def cal_val_loss(self):
        loss, _ = evaluate_model(self.model, self.val_loader, self.device, self.criterion)
        return loss

    def current_privacy_epsilon(self):
        if not self.use_dp:
            return None
        return compute_dp_epsilon(
            sample_rate=self.dp_sample_rate,
            noise_multiplier=self.dp_noise_multiplier,
            steps=self.dp_steps,
            delta=self.dp_delta,
        )

    def fine_tune(self):
        self.model.train()
        for _ in range(self.local_epochs):
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if self.use_dp:
                    selected = dp_sgd_step(
                        model=self.model,
                        optimizer=self.optimizer,
                        criterion=self.criterion,
                        inputs=inputs,
                        labels=labels,
                        max_grad_norm=self.dp_max_grad_norm,
                        noise_multiplier=self.dp_noise_multiplier,
                        sample_rate=self.dp_sample_rate,
                        scheduler=self.scheduler,
                    )
                    if selected > 0:
                        self.dp_steps += 1
                    continue

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

    def test(self, dataloader=None):
        dataloader = self.test_loader if dataloader is None else dataloader
        return evaluate_model(self.model, dataloader, self.device, self.criterion)

    def test_on_all_clients(self, clients):
        return [self.test(client.test_loader)[1] for client in clients]
