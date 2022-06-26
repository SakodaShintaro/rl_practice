import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, h, w, hidden_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convh * convw * 32
        self.head = nn.Linear(linear_input_size, hidden_size)

    def forward(self, x):
        x = x.to(self.conv1.weight.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.head(x)
        return x


class Net(nn.Module):
    def __init__(self, h, w, outputs):
        super(Net, self).__init__()
        self.hidden_size = 256
        self.cnn = CNN(h, w, self.hidden_size)
        # layer = nn.TransformerDecoderLayer(self.hidden_size, 8, dim_feedforward=self.hidden_size * 4, batch_first=True)
        # self.transformer = nn.TransformerDecoder(layer, 2)
        self.linear_p = nn.Linear(self.hidden_size, outputs)
        self.linear_v = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden=None):
        x = x.permute([0, 3, 1, 2])
        x = self.cnn(x)
        p = self.linear_p(x)
        v = self.linear_v(x)
        return {'policy': p, 'value': v}
