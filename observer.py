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
        observation = self.env.reset()

        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.CUBIC),
                                 T.ToTensor()])

        self.state_len = 5
        self.state = deque([self.transform_observation(observation) for _ in range(self.state_len)])
    
    def transform_observation(self, observation):
        x = observation
        x = torch.from_numpy(x)
        x = x.permute([2, 0, 1])
        x = self.resize(x)
        x = x.unsqueeze(0)
        return x

    def get_state(self):
        return torch.cat(tuple(self.state), dim=0).unsqueeze(0)

    def step(self, action):
        result = self.env.step(action)
        observation, reward, done, info = result
        self.state.popleft()
        self.state.append(self.transform_observation(observation))

        # screen = self.transform_observation(observation)
        # screen = screen.cpu().squeeze(0).permute((1, 2, 0)).numpy()
        # plt.imshow(screen, interpolation="none")
        # plt.title("Example extracted screen")
        # plt.savefig(f"images/screen.png", bbox_inches="tight", pad_inches=0.05)
        # plt.cla()
        return result

    def get_n_actions(self):
        return self.env.action_space.n
