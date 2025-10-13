from dataclasses import dataclass

import numpy as np
import torch

""" Note
リプレイバッファのインデックスtにどのタイミングのデータが入っているかは混乱しやすいので記録
（というのも、環境から時刻tとして受け取った後の選択した行動なのか、その前に選択した行動なのか、どちらとペアを取るかは任意性がある）
このコードでは、インデックスtでは、エージェントが時刻tで行動を選択する直前で見える情報を格納するという考えを採る
つまり
- obs, reward, doneはそれぞれ時刻tに環境から得て入力になるもの
- rnn_stateは時刻tでのRNNの隠れ状態
- action、log_prob, valueは時刻t-1で選択した行動およびその対数確率, 価値
を格納する
"""


@dataclass
class ReplayBufferData:
    observations: torch.Tensor  # (B, T, obs_shape)
    rewards: torch.Tensor  # (B, T)
    dones: torch.Tensor  # (B, T)
    rnn_state: torch.Tensor  # (B, hidden_size)
    actions: torch.Tensor  # (B, T, action_shape)
    log_probs: torch.Tensor  # (B, T)
    values: torch.Tensor  # (B, T)


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        seq_len: int,
        obs_shape: np.ndarray,
        rnn_state_shape: np.ndarray,
        action_shape: np.ndarray,
        device: torch.device,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.action_shape = action_shape
        self.device = device

        assert self.seq_len <= self.size, "Replay buffer size must be >= sequence length."

        self.observations = np.zeros((size, *obs_shape), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        self.rnn_states = np.zeros((size, *rnn_state_shape), dtype=np.float32)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.log_probs = np.zeros((size, 1), dtype=np.float32)
        self.values = np.zeros((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def is_full(self) -> bool:
        return self.full

    def reset(self) -> None:
        self.idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        reward: float,
        done: bool,
        rnn_state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
    ) -> None:
        self.observations[self.idx] = obs
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.rnn_states[self.idx] = rnn_state
        self.actions[self.idx] = action
        self.log_probs[self.idx] = log_prob
        self.values[self.idx] = value

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferData:
        curr_size = self.size if self.full else self.idx
        assert curr_size >= self.seq_len, "Not enough data to sample a sequence."

        # Generate base indices for each batch element
        indices = np.random.randint(0, curr_size - self.seq_len, size=batch_size)

        # Create vectorized sequence indices: (batch_size, seq_len)
        seq_indices = indices[:, None] + np.arange(self.seq_len)[None, :]

        # Vectorized slicing - much faster than loop + append
        observations = self.observations[seq_indices]
        rewards = self.rewards[seq_indices]
        dones = self.dones[seq_indices]
        rnn_states = self.rnn_states[seq_indices]
        actions = self.actions[seq_indices]
        log_probs = self.log_probs[seq_indices]
        values = self.values[seq_indices]

        # Use from_numpy instead of tensor() for better performance
        return ReplayBufferData(
            torch.from_numpy(observations).to(self.device),
            torch.from_numpy(rewards).to(self.device),
            torch.from_numpy(dones).to(self.device),
            torch.from_numpy(rnn_states).to(self.device),
            torch.from_numpy(actions).to(self.device),
            torch.from_numpy(log_probs).to(self.device),
            torch.from_numpy(values).to(self.device),
        )

    def get_latest(self, seq_len: int) -> ReplayBufferData:
        # Create vectorized indices for the latest sequence
        indices = (self.idx - seq_len + np.arange(seq_len)) % self.size

        # Vectorized slicing
        observations = self.observations[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        rnn_states = self.rnn_states[indices]
        actions = self.actions[indices]
        log_probs = self.log_probs[indices]
        values = self.values[indices]

        # Use from_numpy and add batch dimension
        return ReplayBufferData(
            torch.from_numpy(observations).unsqueeze(0).to(self.device),
            torch.from_numpy(rewards).unsqueeze(0).to(self.device),
            torch.from_numpy(dones).unsqueeze(0).to(self.device),
            torch.from_numpy(rnn_states).unsqueeze(0).to(self.device),
            torch.from_numpy(actions).unsqueeze(0).to(self.device),
            torch.from_numpy(log_probs).unsqueeze(0).to(self.device),
            torch.from_numpy(values).unsqueeze(0).to(self.device),
        )
