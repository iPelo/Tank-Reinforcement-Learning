from __future__ import annotations

import torch
import torch.nn as nn


HiddenState = tuple[torch.Tensor, torch.Tensor]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, recurrent_hidden: int = 128) -> None:
        super().__init__()

        self.hidden = int(hidden)
        self.recurrent_hidden = int(recurrent_hidden)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(input_size=hidden, hidden_size=recurrent_hidden, num_layers=1)
        self.policy_head = nn.Linear(recurrent_hidden, act_dim)
        self.value_head = nn.Linear(recurrent_hidden, 1)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> HiddenState:
        rnn_device = device if device is not None else self.policy_head.weight.device
        h = torch.zeros(1, batch_size, self.recurrent_hidden, device=rnn_device)
        c = torch.zeros(1, batch_size, self.recurrent_hidden, device=rnn_device)
        return h, c

    def forward(self, obs: torch.Tensor, state: HiddenState | None = None) -> tuple[torch.Tensor, torch.Tensor, HiddenState]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2:
            raise ValueError(f"Expected obs with shape [batch, obs_dim], got {tuple(obs.shape)}")

        batch_size = obs.shape[0]
        if state is None:
            state = self.initial_state(batch_size=batch_size, device=obs.device)

        encoded = self.encoder(obs)
        encoded = encoded.unsqueeze(0)
        rnn_out, next_state = self.rnn(encoded, state)
        features = rnn_out.squeeze(0)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value, next_state

    def evaluate_sequence(
        self,
        obs: torch.Tensor,
        state: HiddenState | None = None,
        episode_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, HiddenState]:
        if obs.ndim != 3:
            raise ValueError(f"Expected obs with shape [batch, time, obs_dim], got {tuple(obs.shape)}")

        batch_size, time_steps, _ = obs.shape
        if state is None:
            state = self.initial_state(batch_size=batch_size, device=obs.device)

        if episode_start is None:
            episode_start = torch.zeros((batch_size, time_steps), dtype=torch.float32, device=obs.device)

        logits_steps: list[torch.Tensor] = []
        value_steps: list[torch.Tensor] = []
        current_state = state

        for step_idx in range(time_steps):
            reset_mask = episode_start[:, step_idx].view(1, batch_size, 1)
            if torch.any(reset_mask > 0):
                current_state = (
                    current_state[0] * (1.0 - reset_mask),
                    current_state[1] * (1.0 - reset_mask),
                )

            step_logits, step_value, current_state = self.forward(obs[:, step_idx], current_state)
            logits_steps.append(step_logits)
            value_steps.append(step_value)

        logits = torch.stack(logits_steps, dim=1)
        value = torch.stack(value_steps, dim=1)
        return logits, value, current_state

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: HiddenState | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HiddenState]:
        logits, value, next_state = self.forward(obs, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, next_state

PolicyModel = ActorCritic | RecurrentActorCritic
