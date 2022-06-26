import gym
import numpy as np
from .net import Net

# stepの結果
# observation, reward, done, info = result
INDEX_OBSERVATION = 0
INDEX_REWARD = 1
INDEX_DONE = 2
INDEX_INFO = 3


class Environment:
    def __init__(self, args={}):
        self.args = args
        self.env = gym.make("Breakout-v4")
        self.curr_step_result = None
        self.sum_reward = 0

    def reset(self, args={}):
        o = self.env.reset()
        self.curr_step_result = [o, 0, False, None]
        self.sum_reward = 0

    def play(self, action, player=None):
        self.curr_step_result = self.env.step(action)

    def step(self, action):
        action = action[0]

        # action0が無限に選ばれ続けると開始されないので
        # 1に無理やり置き換える
        if action == 0:
            action = 1
        self.curr_step_result = self.env.step(action)
        self.sum_reward += self.curr_step_result[INDEX_REWARD]

        # print(self.curr_step_result[1:])
        # print("action = ", action)

        # import matplotlib.pyplot as plt
        # screen = self.curr_step_result[INDEX_OBSERVATION]
        # plt.imshow(screen, interpolation="none")
        # plt.text(0, 0, f"action = {action}")
        # plt.savefig(f"screen.png", bbox_inches="tight", pad_inches=0.05)
        # plt.cla()
        # plt.clf()

    def turns(self):
        return [0]

    def terminal(self):
        return self.curr_step_result[INDEX_DONE]

    def players(self):
        return [0]

    def turn(self):
        return 0

    def reward(self):
        return {0: self.curr_step_result[INDEX_REWARD]}

    def outcome(self):
        print("outcome = ", self.sum_reward)
        return {0: self.sum_reward}

    def legal_actions(self, player=None):
        return list(range(0, self.env.action_space.n))

    def observation(self, player=None):
        return self.curr_step_result[INDEX_OBSERVATION].astype(np.float32)

    def observers(self):
        return [0]

    def net(self):
        return Net(210, 160, 4)

    def action2str(self, action):
        return self.env.get_action_meanings()[action]


if __name__ == '__main__':
    import random
    e = Environment()
    for _ in range(100):
        e.reset()
        while not e.terminal():
            print(e)
            actions = e.legal_actions()
            print([e.action2str(a) for a in actions])
            e.play(random.choice(actions))
            print(e.observation().shape)
        print(e)
        print(e.outcome())
