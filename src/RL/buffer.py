from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import torch

@dataclass
class RolloutBatch:
    obs:torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    adv: torch.Tensor
    rets: torch.Tensor


class RolloutBuffer:
    def __init__(self,size: int, obs_dim: int, device: torch.device )-> None:
        self.size = int(size)
        self.obs_dim = int(obs_dim)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.full = False


        self.obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size, ), dtype=np.int64)
        self.rew = np.zeros((self.size,), dtype=np.float32)
        self.done = np.zeros((self.size,), dtype=np.float32)
        self.logp = np.zeros((self.size,), dtype=np.float32)
        self.val = np.zeros((self.size,), dtype=np.float32)
        self.adv = np.zeros((self.size,), dtype=np.float32)
        self.rets = np.zeros((self.size,), dtype=np.float32)

        def add(self, last_val:float, gamma:float,lam:float) -> None:
            adv = 0.0
            for t in reversed(range(self.size)):
                nonterminal = 1.0 - self.done[t]
                next_val = last_val if t == self.size - 1 else self.val[t+1]
                delta = self.rew[t] + gamma * nonterminal * next_val - self.val[t]
                adv = delta + gamma * lam * nonterminal * adv
                self.adv[t] = adv

            self.ret = self.adv + self.val

        def iter_minibacthes(self, batch_size:int, shuffle: bool = True) -> Iterator[RolloutBatch]:
            idxs = np.arrange(self.size)
            if shuffle:
                np.random.shuffle(idxs)

            for start in range(0, self.size, batch_size):
                mb = idxs[start:start+batch_size]

                obs = torch.as_tensor(self.obs[mb], device=self.device)
                act = torch.as_tensor(self.act[mb], device=self.device)
                logp = torch.as_tensor(self.logp[mb], device=self.device)
                val = torch.as_tensor(self.val[mb], device=self.device)
                adv = torch.as_tensor(self.adv[mb], device=self.device)
                ret = torch.as_tensor(self.ret[mb], device=self.device)

                yield RolloutBatch(obs=obs, act=act, logp=logp, val=val, adv=adv, ret=ret)



