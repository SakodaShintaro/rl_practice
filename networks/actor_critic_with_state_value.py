import numpy as np
import torch
from diffusers.models import AutoencoderTiny
from torch import nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from networks.backbone import RecurrentEncoder, SingleFrameEncoder


class Network1(nn.Module):
    def __init__(self, hidden_size, observation_space, action_space_shape):
        super().__init__()
        self.hidden_size = hidden_size

        # Observation encoder
        self.encoder_type = "simple_cnn"
        if self.encoder_type == "simple_cnn":
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        elif self.encoder_type == "ae":
            self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")

        # Compute output size of convolutional layers
        in_features_next_layer = self._get_conv_output(observation_space.shape)
        print(f"{in_features_next_layer=}")

        self.lin_hidden_in = nn.Linear(in_features_next_layer, self.hidden_size)

        # Recurrent layer (GRU)
        self.recurrent_layer = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # Hidden layer
        self.lin_hidden_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        assert len(action_space_shape) == 1
        self.policy = nn.Linear(in_features=self.hidden_size, out_features=action_space_shape[0])

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)

        # Apply weight initialization to all modules
        self.apply(self._init_weights)

    def _get_conv_output(self, shape: tuple) -> int:
        o = torch.zeros(1, *shape)
        if self.encoder_type == "simple_cnn":
            o = self.conv1(o)
            o = self.conv2(o)
            o = self.conv3(o)
        elif self.encoder_type == "ae":
            o = self.ae.encode(o).latents
        return int(np.prod(o.size()))

    def _init_weights(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if "ae." in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def init_state(self) -> tuple:
        return torch.zeros((1, 1, self.hidden_size))

    def forward(
        self,
        obs: torch.Tensor,  # (B, T, 3, H, W)
        recurrent_cell: torch.Tensor,  #  (1, B, hidden_size)
    ):
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        B, T, C, H, W = h.shape
        h = h.reshape(B * T, C, H, W)
        if self.encoder_type == "simple_cnn":
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
        elif self.encoder_type == "ae":
            with torch.no_grad():
                h = self.ae.encode(h).latents

        h = h.flatten(start_dim=1)
        h = F.relu(self.lin_hidden_in(h))

        # Forward recurrent layer (GRU) first, then hidden layer
        # Reshape the to be fed data to batch_size, sequence_length, data
        h = h.reshape((B, T, -1))

        # Forward recurrent layer
        h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

        # Reshape to the original tensor size
        h = h.reshape(B * T, -1)

        # Feed hidden layer after recurrent layer
        h = F.relu(self.lin_hidden_out(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = Categorical(logits=self.policy(h_policy))

        return pi, value, recurrent_cell


class Network2(nn.Module):
    def __init__(
        self, action_dim: int, seq_len: int, encoder_type: str, image_h: int, image_w: int
    ) -> None:
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_dim = action_dim

        if encoder_type == "recurrent":
            self.encoder = RecurrentEncoder(image_h, image_w)
        else:
            self.encoder = SingleFrameEncoder(seq_len, device)
        seq_hidden_dim = self.encoder.output_dim
        self.linear = nn.Linear(seq_hidden_dim, seq_hidden_dim)
        self.value_enc = nn.Sequential(nn.Linear(seq_hidden_dim, seq_hidden_dim), nn.ReLU())
        self.value_head = nn.Linear(seq_hidden_dim, 1)
        self.policy_enc = nn.Sequential(nn.Linear(seq_hidden_dim, seq_hidden_dim), nn.ReLU())
        self.policy_type = "Categorical"
        if self.policy_type == "Beta":
            self.alpha_head = nn.Linear(seq_hidden_dim, action_dim)
            self.beta_head = nn.Linear(seq_hidden_dim, action_dim)
        elif self.policy_type == "Categorical":
            self.logits_head = nn.Linear(seq_hidden_dim, action_dim)
        else:
            raise ValueError("Invalid policy type")
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with orthogonal initialization.

        Arguments:
            module {nn.Module} -- Module to initialize
        """
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def init_state(self) -> torch.Tensor:
        return self.encoder.init_state()

    def forward(
        self,
        r_seq: torch.Tensor,  # (B, T, 1)
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        a_seq: torch.Tensor,  # (B, T, action_dim)
        rnn_state: torch.Tensor,  # (B, 1, hidden_size)
        action: torch.Tensor | None = None,  # (B, action_dim) or None
    ) -> tuple:
        x, rnn_state = self.encoder(s_seq, a_seq, r_seq, rnn_state)  # (B, T, hidden_dim)
        x = x[:, -1, :]  # (B, hidden_dim)
        x = self.linear(x)  # (B, hidden_dim)

        value_x = self.value_enc(x)
        value = self.value_head(value_x)

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
            dist = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = dist.sample()
                a_logp = dist.log_prob(action).unsqueeze(1)
                action = F.one_hot(action, num_classes=self.action_dim).float()
            else:
                a_logp = dist.log_prob(action.argmax(dim=1)).unsqueeze(1)

        return {
            "action": action,
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "value_x": value_x,
            "policy_x": policy_x,
            "rnn_state": rnn_state,
        }
