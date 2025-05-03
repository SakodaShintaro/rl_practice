import torch
from torch import nn
from torch.distributions import Beta

from .backbone import BaseCNN


class PpoBetaPolicyAndValue(nn.Module):
    def __init__(self, in_channels: int, action_dim: int) -> None:
        super().__init__()
        self.cnn_base = BaseCNN(in_channels)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, action_dim), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m: object) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x: torch.Tensor) -> tuple:
        # x.shape = (batch_size, STACK_SIZE * 3, 96, 96)
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v

    @torch.inference_mode()
    def get_action(self, x: torch.Tensor) -> tuple:
        alpha, beta = self.forward(x)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def calc_action_logp(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.forward(s)[0]
        dist = Beta(alpha, beta)
        a_logp = dist.log_prob(a).sum(dim=1, keepdim=True)
        return a_logp
