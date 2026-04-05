import torch

from .client import Client
from .dp import dp_sgd_step
from .optimizer.fedprox import PerturbedGradientDescent
from .training import create_scheduler, evaluate_model


class ClientProx(Client):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__(args, client_id, model_fn, train_loader, val_loader, test_loader)
        self.mu = args.mu
        if not self.use_dp:
            self.optimizer = PerturbedGradientDescent(self.model.parameters(), lr=args.lr, mu=self.mu)
        total_steps = max(len(self.train_loader) * self.local_epochs * args.global_rounds, 1)
        self.scheduler = create_scheduler(self.optimizer, total_steps)
        self.global_params = [param.detach().clone().to(self.device) for param in self.model.parameters()]

    def set_parameters(self, state_dict):
        super().set_parameters(state_dict)
        self.global_params = [param.detach().clone().to(self.device) for param in self.model.parameters()]

    def fine_tune(self):
        self.model.train()
        for _ in range(self.local_epochs):
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if self.use_dp:
                    def loss_builder(model, sample_input, sample_label, selected_count):
                        logits = model(sample_input)
                        data_loss = self.criterion(logits, sample_label)
                        prox_penalty = 0.0
                        for param, global_param in zip(model.parameters(), self.global_params):
                            prox_penalty = prox_penalty + 0.5 * self.mu * torch.sum((param - global_param) ** 2)
                        return data_loss + prox_penalty / max(selected_count, 1)

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
                        loss_builder=loss_builder,
                    )
                    if selected > 0:
                        self.dp_steps += 1
                    continue

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step(self.global_params, self.device)
                if self.scheduler is not None:
                    self.scheduler.step()
