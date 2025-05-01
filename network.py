import math

import gymnasium as gym
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

RESNET_DIM = 512
RESNET_DIR = "./resnet18"


class BaseCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=4, stride=2),  # -> (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # -> (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # -> (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # -> (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # -> (128, 3, 3)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # -> (256, 1, 1)
            nn.ReLU(),
            nn.Flatten(),  # -> (256,)
        )

    def forward(self, x):
        return self.features(x)


def orthogonal_weight_init(m: nn.Module) -> None:
    """Orthogonal weight initialization for neural networks."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class SoftQNetwork(nn.Module):
    def __init__(self, env: gym.Env, hidden_dim: int, use_normalize: bool = True) -> None:
        super().__init__()
        input_dim = RESNET_DIM + np.prod(env.action_space.shape)
        self.resnet = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, cache_dir=RESNET_DIR
        )
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.use_normalize = use_normalize
        self.apply(orthogonal_weight_init)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # (bs, h, w, c) -> (bs, c, h, w)
        x = x / 255.0
        x = self.resnet(x)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.use_normalize:
            x = x / torch.norm(x, dim=1).view((-1, 1))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env: gym.Env, hidden_dim: int, use_normalize: bool = True) -> None:
        super().__init__()
        self.resnet = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, cache_dir=RESNET_DIR
        )
        self.fc1 = nn.Linear(RESNET_DIM, hidden_dim)
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 3, 1, 2)  # (bs, h, w, c) -> (bs, c, h, w)
        x = x / 255.0
        x = self.resnet(x)
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

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class DiffusionActor(nn.Module):
    def __init__(self, env: gym.Env, use_normalize: bool = False) -> None:
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

    def forward(self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        t = self.t_embedder(t)
        a = torch.cat([a, t, state], 1)
        a = F.relu(self.fc1(a))
        a = F.relu(self.fc2(a))
        if self.use_normalize:
            a = a / torch.norm(a, dim=1).view((-1, 1))
        a = self.fc3(a)
        return a

    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
