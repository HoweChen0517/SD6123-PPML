import copy

import torch

from .client import Client
from .dp import compute_dp_epsilon, dp_sgd_step
from .optimizer.fedprox import PerturbedGradientDescent
from .training import create_scheduler, evaluate_model


class ClientDitto(Client):
    def __init__(self, args, client_id, model_fn, train_loader, val_loader, test_loader):
        super().__init__(args, client_id, model_fn, train_loader, val_loader, test_loader)
        self.mu = args.mu
        self.plocal_epochs = args.plocal_epochs
        self.personal_model = copy.deepcopy(self.model).to(self.device)
        if self.use_dp:
            self.personal_optimizer = torch.optim.SGD(
                self.personal_model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            self.personal_optimizer = PerturbedGradientDescent(
                self.personal_model.parameters(), lr=args.lr, mu=self.mu
            )
        total_steps = max(len(self.train_loader) * max(self.plocal_epochs, 1) * args.global_rounds, 1)
        self.personal_scheduler = create_scheduler(self.personal_optimizer, total_steps)
        self.global_params = [param.detach().clone().to(self.device) for param in self.model.parameters()]
        self.personal_initialized = False
        self.personal_dp_steps = 0

    def set_parameters(self, state_dict):
        super().set_parameters(state_dict)
        self.global_params = [param.detach().clone().to(self.device) for param in self.model.parameters()]
        if not self.personal_initialized:
            self.personal_model.load_state_dict(self.model.state_dict())
            self.personal_initialized = True

    def p_fine_tune(self):
        self.personal_model.train()
        for _ in range(self.plocal_epochs):
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
                        model=self.personal_model,
                        optimizer=self.personal_optimizer,
                        criterion=self.criterion,
                        inputs=inputs,
                        labels=labels,
                        max_grad_norm=self.dp_max_grad_norm,
                        noise_multiplier=self.dp_noise_multiplier,
                        sample_rate=self.dp_sample_rate,
                        scheduler=self.personal_scheduler,
                        loss_builder=loss_builder,
                    )
                    if selected > 0:
                        self.personal_dp_steps += 1
                    continue

                logits = self.personal_model(inputs)
                loss = self.criterion(logits, labels)
                self.personal_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.personal_model.parameters(), self.grad_clip)
                self.personal_optimizer.step(self.global_params, self.device)
                if self.personal_scheduler is not None:
                    self.personal_scheduler.step()

    def current_personal_privacy_epsilon(self):
        if not self.use_dp:
            return None
        return compute_dp_epsilon(
            sample_rate=self.dp_sample_rate,
            noise_multiplier=self.dp_noise_multiplier,
            steps=self.personal_dp_steps,
            delta=self.dp_delta,
        )

    def test(self, dataloader=None, personalized=False):
        dataloader = self.test_loader if dataloader is None else dataloader
        model = self.personal_model if personalized else self.model
        return evaluate_model(model, dataloader, self.device, self.criterion)

    def cal_val_loss(self):
        return self.test(self.val_loader, personalized=True)[0]

    def test_on_all_clients(self, clients):
        return [self.test(client.test_loader, personalized=True)[1] for client in clients]
