import torch
from torch import nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from networks.backbone import RecurrentEncoder
from networks.value_head import StateValueHead


class Network(nn.Module):
    def __init__(
        self, observation_space_shape: list[int], action_space_shape: list[int], num_bins: int
    ) -> None:
        super().__init__()
        self.action_dim = action_space_shape[0]
        self.encoder = RecurrentEncoder(observation_space_shape)
        hidden_dim = self.encoder.output_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = StateValueHead(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            block_num=1,
            num_bins=num_bins,
            sparsity=0.0,
        )
        self.policy_enc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.policy_type = "Categorical"
        if self.policy_type == "Beta":
            self.alpha_head = nn.Linear(hidden_dim, self.action_dim)
            self.beta_head = nn.Linear(hidden_dim, self.action_dim)
        elif self.policy_type == "Categorical":
            self.logits_head = nn.Linear(hidden_dim, self.action_dim)
        else:
            raise ValueError("Invalid policy type")
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with orthogonal initialization.

        Arguments:
            module {nn.Module} -- Module to initialize
        """
        for name, param in module.named_parameters():
            if "ae." in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def forward(
        self,
        r_seq: torch.Tensor,  # (B, T, 1)
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        a_seq: torch.Tensor,  # (B, T, action_dim)
        rnn_state: torch.Tensor,  # (1, B, hidden_size)
        action: torch.Tensor | None = None,  # (B, action_dim) or None
    ) -> tuple:
        x, rnn_state = self.encoder(s_seq, a_seq, r_seq, rnn_state)  # (B, T, hidden_dim)
        x = x[:, -1, :]  # (B, hidden_dim)
        x = self.linear(x)  # (B, hidden_dim)

        value_dict = self.value_head(x)

        policy_x = self.policy_enc(x)

        if self.policy_type == "Beta":
            alpha = self.alpha_head(policy_x).exp() + 1
            beta = self.beta_head(policy_x).exp() + 1

            dist = Beta(alpha, beta)
            if action is None:
                action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1, keepdim=True)
        elif self.policy_type == "Categorical":
            logits = self.logits_head(policy_x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
                a_logp = dist.log_prob(action).unsqueeze(1)
                action = F.one_hot(action, num_classes=self.action_dim).float()
            else:
                a_logp = dist.log_prob(action.argmax(dim=1)).unsqueeze(1)

        return {
            "action": action,  # (B, action_dim)
            "a_logp": a_logp,  # (B, 1)
            "entropy": dist.entropy().unsqueeze(1),  # (B, 1)
            "value": value_dict["output"],  # (B, 1)
            "x": x,  # (B, hidden_dim)
            "rnn_state": rnn_state,  # (1, B, hidden_size)
        }
