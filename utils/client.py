import copy

import torch
from torch import nn

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
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

    def test(self, dataloader=None):
        dataloader = self.test_loader if dataloader is None else dataloader
        return evaluate_model(self.model, dataloader, self.device, self.criterion)

    def test_on_all_clients(self, clients):
        return [self.test(client.test_loader)[1] for client in clients]
