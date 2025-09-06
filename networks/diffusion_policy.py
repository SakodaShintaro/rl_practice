import math

import torch
from torch import nn

from .blocks import SimbaBlock
from .sparse_utils import apply_one_shot_pruning


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
        hidden_dim: int,
        block_num: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        time_embedding_size = 256
        self.fc_in = nn.Linear(state_dim + action_dim + time_embedding_size, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.step_num = 5
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(
        self, a: torch.Tensor, t: torch.Tensor, state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        result_dict = {}

        t = self.t_embedder(t)
        x = torch.cat([a, t, state], 1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        result_dict["activation"] = x

        x = self.fc_out(x)
        result_dict["output"] = x
        return result_dict

    def get_action(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = x.size(0)
        normal = torch.distributions.Normal(
            torch.zeros((bs, self.action_dim), device=x.device),
            torch.ones((bs, self.action_dim), device=x.device),
        )
        action = normal.sample().to(x.device)
        action = torch.clamp(action, -3.0, 3.0)
        dt = 1.0 / self.step_num

        curr_time = torch.zeros((bs), device=x.device)

        for _ in range(self.step_num):
            tmp_dict = self.forward(action, curr_time, x)
            v = tmp_dict["output"]
            action = action + dt * v
            curr_time += dt

        action = torch.tanh(action)

        # nanがあったら終了
        if torch.isnan(action).any():
            print(f"{action=}")
            print(f"{self.fc_in.weight=}")
            import sys

            sys.exit(1)

        dummy_log_p = torch.zeros((bs, 1), device=x.device)
        return action, dummy_log_p


class DiffusionStatePredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        hidden_dim: int,
        block_num: int,
        sparsity: float,
    ) -> None:
        super().__init__()
        time_embedding_size = 256
        self.fc_in = nn.Linear(input_dim + state_dim + time_embedding_size, hidden_dim)
        self.fc_mid = nn.Sequential(*[SimbaBlock(hidden_dim) for _ in range(block_num)])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc_out = nn.Linear(hidden_dim, state_dim)
        self.state_dim = state_dim
        self.step_num = 50
        self.t_embedder = TimestepEmbedder(time_embedding_size)
        self.sparse_mask = (
            None if sparsity == 0.0 else apply_one_shot_pruning(self, overall_sparsity=sparsity)
        )

    def forward(
        self, target: torch.Tensor, t: torch.Tensor, input_token: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        result_dict = {}

        t = self.t_embedder(t)
        x = torch.cat([target, t, input_token], 1)
        x = self.fc_in(x)

        x = self.fc_mid(x)
        x = self.norm(x)

        result_dict["activation"] = x

        x = self.fc_out(x)
        result_dict["output"] = x
        return result_dict

    def sample(self, input_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = input_token.size(0)
        normal = torch.distributions.Normal(
            torch.zeros((bs, self.state_dim), device=input_token.device),
            torch.ones((bs, self.state_dim), device=input_token.device),
        )
        target = normal.sample().to(input_token.device)
        target = torch.clamp(target, -3.0, 3.0)
        dt = 1.0 / self.step_num

        curr_time = torch.zeros((bs), device=input_token.device)

        for _ in range(self.step_num):
            tmp_dict = self.forward(target, curr_time, input_token)
            v = tmp_dict["output"]
            target = target + dt * v
            curr_time += dt

        if torch.isnan(target).any():
            print(f"{target=}")
            print(f"{self.fc_in.weight=}")
            import sys

            sys.exit(1)

        dummy_log_p = torch.zeros((bs, 1), device=input_token.device)
        return target, dummy_log_p
