from __future__ import annotations

from dataclasses import dataclass

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

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: HiddenState | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, HiddenState]:
        logits, value, next_state = self.forward(obs, state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, next_state


@dataclass
class ModelIO:
    obs_dim: int
    act_dim: int
    hidden: int = 128

    def build(self) -> ActorCritic:
        return ActorCritic(self.obs_dim, self.act_dim, self.hidden)


@dataclass
class RecurrentModelIO:
    obs_dim: int
    act_dim: int
    hidden: int = 128
    recurrent_hidden: int = 128

    def build(self) -> RecurrentActorCritic:
        return RecurrentActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden=self.hidden,
            recurrent_hidden=self.recurrent_hidden,
        )
