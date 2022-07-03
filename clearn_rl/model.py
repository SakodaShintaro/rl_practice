import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        hidden_ch = 512
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, hidden_ch),
            nn.ReLU(),
        )

        self.pe = PositionalEncoding(hidden_ch)
        layer = nn.TransformerEncoderLayer(hidden_ch, 8, hidden_ch * 4, batch_first=True)
        self.trans = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(hidden_ch, env.single_action_space.n)

    def forward(self, x):
        x = x / 255.0
        x_list = list()
        for ch in range(x.shape[1]):
            curr_x = x[:, ch:ch + 1, :, :]
            curr_x = self.cnn(curr_x)
            curr_x = curr_x.unsqueeze(1)
            x_list.append(curr_x)
        x = torch.cat(x_list, dim=1)
        x = self.pe(x)
        x = self.trans(x)
        x = x[:, -1, :]
        x = self.head(x)
        return x
