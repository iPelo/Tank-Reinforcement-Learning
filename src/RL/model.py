from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim:int,act_dim:int,hidden: int = 128) -> None :
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

        def foward(self,obs:torch.Tensor):
            x = self.backbone(obs)
            logits = self.policy_head(x)
            value = self.value_head(x).squeeze(-1)
            return logits,value

        @torch.no_grad()
        def act(self,obs: torch.Tensor):
            logits, value = self.fowards(obs)
            return logits,value
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            return action,logp, value

        @dataclass
        class ModelIO:
            obs_dim: int
            act_dim: int
            hidden: int

            def build(self) -> ActorCritic:
                return ActorCritic(self.obs_dim,self.act_dim,self.hidden)

