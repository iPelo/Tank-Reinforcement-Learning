from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .model import ActorCritic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5

    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 8


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        adv = torch.as_tensor(buf.adv, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf.adv[:] = adv.detach().cpu().numpy()

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        total_kl = 0.0
        n = 0

        for _ in range(self.cfg.update_epochs):
            for batch in buf.iter_minibatches(self.cfg.minibatch_size, shuffle=True):
                logits, value = self.model(batch.obs)
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(batch.act)
                ratio = torch.exp(logp - batch.logp)

                surr1 = ratio * batch.adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * batch.adv
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = 0.5 * (batch.ret - value).pow(2).mean()
                ent = dist.entropy().mean()

                loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (batch.logp - logp).mean().abs().item()

                total_pi_loss += float(pi_loss.item())
                total_v_loss += float(v_loss.item())
                total_ent += float(ent.item())
                total_kl += float(approx_kl)
                n += 1

        return {
            "pi_loss": total_pi_loss / max(1, n),
            "v_loss": total_v_loss / max(1, n),
            "entropy": total_ent / max(1, n),
            "approx_kl": total_kl / max(1, n),
        }