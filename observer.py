from collections import deque
import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


class Observer:
    def __init__(self):
        self.env = gym.make("Breakout-v4")
        self.env.reset()

        self.resize = T.Compose([T.ToPILImage(),
                                 T.Resize(40, interpolation=Image.CUBIC),
                                 T.ToTensor()])

        self.state_len = 5
        self.state = deque([self.get_curr_screen() for _ in range(self.state_len)])

    def get_curr_screen(self):
        screen = self.env.render(mode="rgb_array").transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        screen = self.resize(screen).unsqueeze(0)
        return screen
    
    def get_state(self):
        return torch.cat(tuple(self.state), dim=0).unsqueeze(0)

    def step(self, action):
        result = self.env.step(action)
        self.state.popleft()
        self.state.append(self.get_curr_screen())
        # screen = self.get_curr_screen()
        # plot_screen = screen.cpu().squeeze(0).permute((1, 2, 0)).numpy()
        # plt.imshow(plot_screen, interpolation="none")
        # plt.title("Example extracted screen")
        # plt.savefig(f"images/screen.png", bbox_inches="tight", pad_inches=0.05)
        # plt.cla()
        return result

    def get_n_actions(self):
        return self.env.action_space.n
