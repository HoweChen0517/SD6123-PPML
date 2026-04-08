from .opacus_client import OpacusClient
from .opacus_dp import apply_proximal_step, clone_trainable_parameters


class OpacusClientProx(OpacusClient):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__(args, client_id, model_fn, train_loader, val_loader, test_loader)
        self.mu = args.mu
        self.global_params = clone_trainable_parameters(self.model)

    def set_parameters(self, state_dict):
        super().set_parameters(state_dict)
        self.global_params = clone_trainable_parameters(self.model)

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
                apply_proximal_step(self.model, self.global_params, self.mu, self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.dp_steps += 1
