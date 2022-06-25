from collections import deque
import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80


class Observer:
    def __init__(self):
        self.env = gym.make("Breakout-v4")
        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.CUBIC),
                                 T.ToTensor()])

        self.state_len = 5
        self.state = deque([torch.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)) for _ in range(self.state_len)])
        self.reset()

    def transform_observation(self, observation):
        x = observation
        x = torch.from_numpy(x)
        x = x.permute([2, 0, 1])
        x = self.resize(x)
        x = x.unsqueeze(0)
        return x

    def get_state(self):
        # for i in range(self.state_len):
        #     ax = plt.subplot(1, self.state_len, i + 1)
        #     screen = self.state[i]
        #     screen = screen.cpu().squeeze(0).permute((1, 2, 0)).numpy()
        #     ax.imshow(screen, interpolation="none")
        # plt.savefig(f"images/screen.png", bbox_inches="tight", pad_inches=0.05)
        # plt.cla()

        return torch.cat(tuple(self.state), dim=0).unsqueeze(0)

    def step(self, action):
        result = self.env.step(action)
        observation, reward, done, info = result
        self.state.popleft()
        self.state.append(self.transform_observation(observation))
        return result

    def get_n_actions(self):
        return self.env.action_space.n

    def reset(self):
        # resetを示す黒画面を一回挿入
        self.state.popleft()
        self.state.append(torch.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)))

        # reset後の状態を取得
        observation = self.env.reset()
        self.state.popleft()
        self.state.append(self.transform_observation(observation))
