import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def orthogonal_weight_init(m: nn.Module) -> None:
    """Orthogonal weight initialization for neural networks."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim, use_normalize: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape),
            hidden_dim,
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.use_normalize = use_normalize
        self.apply(orthogonal_weight_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, hidden_dim, use_normalize: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.use_normalize = use_normalize
        self.apply(orthogonal_weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000) -> torch.Tensor:
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

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiffusionActor(nn.Module):
    def __init__(self, env, use_normalize: bool = False):
        super().__init__()
        time_embedding_size = 256
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape)
            + time_embedding_size,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.use_normalize = use_normalize
        self.action_dim = np.prod(env.action_space.shape)
        self.step_num = 5
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.apply(orthogonal_weight_init)

    def forward(self, a, t, state):
        t = self.t_embedder(t)
        a = torch.cat([a, t, state], 1)
        a = F.relu(self.fc1(a))
        a = F.relu(self.fc2(a))
        if self.use_normalize:
            a = a / torch.norm(a, dim=1).view((-1, 1))
        a = self.fc3(a)
        return a

    def get_action(self, x):
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

        return action, torch.zeros((1, 1)), torch.Tensor([0.0])
