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
        indices = np.random.randint(0, curr_size - self.seq_len, size=batch_size)
        observations = []
        rewards = []
        dones = []
        rnn_states = []
        actions = []
        log_probs = []
        values = []
        for idx in indices:
            observations.append(self.observations[idx : idx + self.seq_len])
            rewards.append(self.rewards[idx : idx + self.seq_len])
            dones.append(self.dones[idx : idx + self.seq_len])
            rnn_states.append(self.rnn_states[idx : idx + self.seq_len])
            actions.append(self.actions[idx : idx + self.seq_len])
            log_probs.append(self.log_probs[idx : idx + self.seq_len])
            values.append(self.values[idx : idx + self.seq_len])
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        rnn_states = np.stack(rnn_states)
        actions = np.stack(actions)
        log_probs = np.stack(log_probs)
        values = np.stack(values)
        return ReplayBufferData(
            torch.tensor(observations).to(self.device),
            torch.tensor(rewards).to(self.device),
            torch.tensor(dones).to(self.device),
            torch.tensor(rnn_states).to(self.device),
            torch.tensor(actions).to(self.device),
            torch.tensor(log_probs).to(self.device),
            torch.tensor(values).to(self.device),
        )

    def get_latest(self, seq_len: int) -> ReplayBufferData:
        observations = []
        rewards = []
        dones = []
        rnn_states = []
        actions = []
        log_probs = []
        values = []
        for i in range(seq_len):
            idx = (self.idx - seq_len + i) % self.size
            observations.append(self.observations[idx])
            rewards.append(self.rewards[idx])
            dones.append(self.dones[idx])
            rnn_states.append(self.rnn_states[idx])
            actions.append(self.actions[idx])
            log_probs.append(self.log_probs[idx])
            values.append(self.values[idx])

        observations = np.stack(observations)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        rnn_states = np.stack(rnn_states)
        actions = np.stack(actions)
        log_probs = np.stack(log_probs)
        values = np.stack(values)
        return ReplayBufferData(
            torch.tensor(observations).unsqueeze(0).to(self.device),
            torch.tensor(rewards).unsqueeze(0).to(self.device),
            torch.tensor(dones).unsqueeze(0).to(self.device),
            torch.tensor(rnn_states).unsqueeze(0).to(self.device),
            torch.tensor(actions).unsqueeze(0).to(self.device),
            torch.tensor(log_probs).unsqueeze(0).to(self.device),
            torch.tensor(values).unsqueeze(0).to(self.device),
        )
