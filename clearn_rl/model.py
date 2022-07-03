import torch
import torch.nn as nn

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

        self.pe = nn.Parameter(torch.zeros([1, 4, hidden_ch]), requires_grad=True)
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
        x = x + self.pe
        x = self.trans(x)
        x = x[:, -1, :]
        x = self.head(x)
        return x
