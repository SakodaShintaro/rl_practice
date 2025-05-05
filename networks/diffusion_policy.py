import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: np.ndarray,
        action_bias: np.ndarray,
        use_normalize: bool = False,
    ) -> None:
        super().__init__()
        time_embedding_size = 256
        self.fc1 = nn.Linear(state_dim + action_dim + time_embedding_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(action_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(action_bias, dtype=torch.float32),
        )
        self.use_normalize = use_normalize
        self.action_dim = action_dim
        self.step_num = 5
        self.t_embedder = TimestepEmbedder(time_embedding_size)

    def forward(self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        t = self.t_embedder(t)
        a = torch.cat([a, t, state], 1)
        a = F.relu(self.fc1(a))
        a = F.relu(self.fc2(a))
        if self.use_normalize:
            a = a / torch.norm(a, dim=1).view((-1, 1))
        a = self.fc3(a)
        return a

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = x.size(0)
        normal = torch.distributions.Normal(
            torch.zeros((bs, self.action_dim), device=x.device),
            torch.ones((bs, self.action_dim), device=x.device),
        )
        a_t = normal.sample().to(x.device)
        log_prob = normal.log_prob(a_t)
        dt = 1.0 / self.step_num

        curr_time = torch.zeros((bs), device=x.device)

        for _ in range(self.step_num):
            v = self.forward(a_t, curr_time, x)
            a_t = a_t + dt * v
            curr_time += dt

        y_t = torch.tanh(a_t)

        action = y_t * self.action_scale + self.action_bias
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        # nanがあったら終了
        if torch.isnan(action).any():
            print(f"{action=}")
            print(f"{self.fc1.weight=}")
            import sys

            sys.exit(1)

        return action, torch.zeros((1, 1))
