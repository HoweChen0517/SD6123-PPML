from torch import nn

from .opacus_dp import clone_model_state, load_model_state, make_private_components, safe_get_epsilon
from .training import evaluate_model


class OpacusClient(nn.Module):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__()
        self.args = args
        self.id = client_id
        self.device = args.device
        self.local_epochs = args.local_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dp_noise_multiplier = args.dp_noise_multiplier
        self.dp_max_grad_norm = args.dp_max_grad_norm
        default_delta = 1.0 / max(len(self.train_loader.dataset), 1)
        self.dp_delta = getattr(args, "dp_delta", default_delta)
        (
            self.model,
            self.optimizer,
            self.criterion,
            self.private_train_loader,
            self.scheduler,
            self.privacy_engine,
            self.dp_sample_rate,
        ) = make_private_components(args, model_fn(), train_loader, self.local_epochs)
        self.dp_steps = 0

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    def set_parameters(self, state_dict):
        load_model_state(self.model, state_dict, self.device)

    def get_parameters(self):
        return clone_model_state(self.model)

    def cal_val_loss(self):
        loss, _ = evaluate_model(self.model, self.val_loader, self.device, self.criterion)
        return loss

    def current_privacy_epsilon(self):
        return safe_get_epsilon(self.privacy_engine, self.dp_delta, self.dp_steps)

    def fine_tune(self):
        self.model.train()
        for _ in range(self.local_epochs):
            for inputs, labels in self.private_train_loader:
                if labels.numel() == 0:
                    continue
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.dp_steps += 1

    def test(self, dataloader=None):
        dataloader = self.test_loader if dataloader is None else dataloader
        return evaluate_model(self.model, dataloader, self.device, self.criterion)

    def test_on_all_clients(self, clients):
        return [self.test(client.test_loader)[1] for client in clients]
