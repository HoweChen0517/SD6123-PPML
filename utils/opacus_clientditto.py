import copy

from .opacus_client import OpacusClient
from .opacus_dp import (
    apply_proximal_step,
    clone_trainable_parameters,
    make_private_components,
    safe_get_epsilon,
)
from .training import evaluate_model


class OpacusClientDitto(OpacusClient):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__(args, client_id, model_fn, train_loader, val_loader, test_loader)
        self.mu = args.mu
        self.plocal_epochs = args.plocal_epochs
        (
            self.personal_model,
            self.personal_optimizer,
            _,
            self.personal_private_train_loader,
            self.personal_scheduler,
            self.personal_privacy_engine,
            self.personal_dp_sample_rate,
        ) = make_private_components(args, model_fn(), train_loader, self.plocal_epochs)
        self.personal_model.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=False)
        self.global_params = clone_trainable_parameters(self.model)
        self.personal_initialized = True
        self.personal_dp_steps = 0

    def set_parameters(self, state_dict):
        super().set_parameters(state_dict)
        self.global_params = clone_trainable_parameters(self.model)
        if not self.personal_initialized:
            self.personal_model.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=False)
            self.personal_initialized = True

    def p_fine_tune(self):
        self.personal_model.train()
        for _ in range(self.plocal_epochs):
            for inputs, labels in self.personal_private_train_loader:
                if labels.numel() == 0:
                    continue
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.personal_optimizer.zero_grad()
                logits = self.personal_model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.personal_optimizer.step()
                apply_proximal_step(self.personal_model, self.global_params, self.mu, self.personal_optimizer)
                if self.personal_scheduler is not None:
                    self.personal_scheduler.step()
                self.personal_dp_steps += 1

    def current_personal_privacy_epsilon(self):
        return safe_get_epsilon(self.personal_privacy_engine, self.dp_delta, self.personal_dp_steps)

    def test(self, dataloader=None, personalized=False):
        dataloader = self.test_loader if dataloader is None else dataloader
        model = self.personal_model if personalized else self.model
        return evaluate_model(model, dataloader, self.device, self.criterion)

    def cal_val_loss(self):
        return self.test(self.val_loader, personalized=True)[0]

    def test_on_all_clients(self, clients):
        return [self.test(client.test_loader, personalized=True)[1] for client in clients]
