import torch

from .client import Client
from .optimizer.fedprox import PerturbedGradientDescent
from .training import create_scheduler, evaluate_model


class ClientProx(Client):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__(args, client_id, model_fn, train_loader, val_loader, test_loader)
        self.mu = args.mu
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
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step(self.global_params, self.device)
                if self.scheduler is not None:
                    self.scheduler.step()
