import torch.nn as nn


def init_weights(m, method: str):
    if isinstance(m, nn.Linear):
        if method == "xavier":
            nn.init.xavier_uniform_(m.weight, gain=1)
        elif method == "orthogonal":
            nn.init.orthogonal_(m.weight.data)
        elif method == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

        nn.init.constant_(m.bias, 0)
