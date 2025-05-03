from torch import nn


class BaseCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # input shape: (batch_size, in_channels, 96, 96)
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
