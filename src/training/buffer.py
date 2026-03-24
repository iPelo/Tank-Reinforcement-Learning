from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor


@dataclass
class RecurrentRolloutBatch:
    obs: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor
    done: torch.Tensor
    episode_start: torch.Tensor
    mask: torch.Tensor
    init_h: torch.Tensor
    init_c: torch.Tensor


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, device: torch.device) -> None:
        self.size = int(size)
        self.obs_dim = int(obs_dim)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size,), dtype=np.int64)
        self.rew = np.zeros((self.size,), dtype=np.float32)
        self.done = np.zeros((self.size,), dtype=np.float32)
        self.logp = np.zeros((self.size,), dtype=np.float32)
        self.val = np.zeros((self.size,), dtype=np.float32)

        self.adv = np.zeros((self.size,), dtype=np.float32)
        self.ret = np.zeros((self.size,), dtype=np.float32)

    def add(self, obs, act, rew, done, logp, val) -> None:
        if self.ptr >= self.size:
            raise RuntimeError("RolloutBuffer overflow")

        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = float(done)
        self.logp[self.ptr] = logp
        self.val[self.ptr] = val

        self.ptr += 1
        if self.ptr == self.size:
            self.full = True

    def compute_gae(self, last_val: float, gamma: float, lam: float) -> None:
        adv = 0.0
        for t in reversed(range(self.size)):
            nonterminal = 1.0 - self.done[t]
            next_val = last_val if t == self.size - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * nonterminal * next_val - self.val[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.adv[t] = adv

        self.ret = self.adv + self.val

    def iter_minibatches(self, batch_size: int, shuffle: bool = True) -> Iterator[RolloutBatch]:
        idxs = np.arange(self.size)
        if shuffle:
            np.random.shuffle(idxs)

        for start in range(0, self.size, batch_size):
            mb = idxs[start : start + batch_size]

            obs = torch.as_tensor(self.obs[mb], device=self.device)
            act = torch.as_tensor(self.act[mb], device=self.device)
            logp = torch.as_tensor(self.logp[mb], device=self.device)
            val = torch.as_tensor(self.val[mb], device=self.device)
            adv = torch.as_tensor(self.adv[mb], device=self.device)
            ret = torch.as_tensor(self.ret[mb], device=self.device)

            yield RolloutBatch(obs=obs, act=act, logp=logp, val=val, adv=adv, ret=ret)


class RecurrentRolloutBuffer:
    def __init__(self, size: int, obs_dim: int, recurrent_hidden: int, device: torch.device) -> None:
        self.size = int(size)
        self.obs_dim = int(obs_dim)
        self.recurrent_hidden = int(recurrent_hidden)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.size,), dtype=np.int64)
        self.rew = np.zeros((self.size,), dtype=np.float32)
        self.done = np.zeros((self.size,), dtype=np.float32)
        self.episode_start = np.zeros((self.size,), dtype=np.float32)
        self.logp = np.zeros((self.size,), dtype=np.float32)
        self.val = np.zeros((self.size,), dtype=np.float32)
        self.init_h = np.zeros((self.size, self.recurrent_hidden), dtype=np.float32)
        self.init_c = np.zeros((self.size, self.recurrent_hidden), dtype=np.float32)

        self.adv = np.zeros((self.size,), dtype=np.float32)
        self.ret = np.zeros((self.size,), dtype=np.float32)

    def add(self, obs, act, rew, done, episode_start, logp, val, state) -> None:
        if self.ptr >= self.size:
            raise RuntimeError("RecurrentRolloutBuffer overflow")

        h, c = state
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = float(done)
        self.episode_start[self.ptr] = float(episode_start)
        self.logp[self.ptr] = logp
        self.val[self.ptr] = val
        self.init_h[self.ptr] = h[0, 0].detach().cpu().numpy()
        self.init_c[self.ptr] = c[0, 0].detach().cpu().numpy()

        self.ptr += 1
        if self.ptr == self.size:
            self.full = True

    def compute_gae(self, last_val: float, gamma: float, lam: float) -> None:
        adv = 0.0
        for t in reversed(range(self.size)):
            nonterminal = 1.0 - self.done[t]
            next_val = last_val if t == self.size - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * nonterminal * next_val - self.val[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.adv[t] = adv

        self.ret = self.adv + self.val

    def iter_sequence_minibatches(
        self,
        sequence_length: int,
        sequences_per_batch: int,
        shuffle: bool = True,
    ) -> Iterator[RecurrentRolloutBatch]:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if sequences_per_batch <= 0:
            raise ValueError("sequences_per_batch must be > 0")

        starts = np.arange(0, self.size, sequence_length)
        if shuffle:
            np.random.shuffle(starts)

        for start_idx in range(0, len(starts), sequences_per_batch):
            batch_starts = starts[start_idx : start_idx + sequences_per_batch]
            batch_size = len(batch_starts)

            obs = np.zeros((batch_size, sequence_length, self.obs_dim), dtype=np.float32)
            act = np.zeros((batch_size, sequence_length), dtype=np.int64)
            logp = np.zeros((batch_size, sequence_length), dtype=np.float32)
            val = np.zeros((batch_size, sequence_length), dtype=np.float32)
            adv = np.zeros((batch_size, sequence_length), dtype=np.float32)
            ret = np.zeros((batch_size, sequence_length), dtype=np.float32)
            done = np.zeros((batch_size, sequence_length), dtype=np.float32)
            episode_start = np.zeros((batch_size, sequence_length), dtype=np.float32)
            mask = np.zeros((batch_size, sequence_length), dtype=np.float32)
            init_h = np.zeros((1, batch_size, self.recurrent_hidden), dtype=np.float32)
            init_c = np.zeros((1, batch_size, self.recurrent_hidden), dtype=np.float32)

            for batch_row, seq_start in enumerate(batch_starts):
                seq_end = min(seq_start + sequence_length, self.size)
                seq_len = seq_end - seq_start

                obs[batch_row, :seq_len] = self.obs[seq_start:seq_end]
                act[batch_row, :seq_len] = self.act[seq_start:seq_end]
                logp[batch_row, :seq_len] = self.logp[seq_start:seq_end]
                val[batch_row, :seq_len] = self.val[seq_start:seq_end]
                adv[batch_row, :seq_len] = self.adv[seq_start:seq_end]
                ret[batch_row, :seq_len] = self.ret[seq_start:seq_end]
                done[batch_row, :seq_len] = self.done[seq_start:seq_end]
                episode_start[batch_row, :seq_len] = self.episode_start[seq_start:seq_end]
                mask[batch_row, :seq_len] = 1.0

                # Store the hidden state for the first element in each sequence chunk.
                init_h[0, batch_row] = self.init_h[seq_start]
                init_c[0, batch_row] = self.init_c[seq_start]

            yield RecurrentRolloutBatch(
                obs=torch.as_tensor(obs, device=self.device),
                act=torch.as_tensor(act, device=self.device),
                logp=torch.as_tensor(logp, device=self.device),
                val=torch.as_tensor(val, device=self.device),
                adv=torch.as_tensor(adv, device=self.device),
                ret=torch.as_tensor(ret, device=self.device),
                done=torch.as_tensor(done, device=self.device),
                episode_start=torch.as_tensor(episode_start, device=self.device),
                mask=torch.as_tensor(mask, device=self.device),
                init_h=torch.as_tensor(init_h, device=self.device),
                init_c=torch.as_tensor(init_c, device=self.device),
            )
