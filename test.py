import cv2
import gym
import random

ENV_NAME = 'Breakout-v4'
NUM_EPISODES = 120
SAVE_DIR = "./images"


class Agent():
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions

    def get_action(self, state):
        action = random.randrange(self.num_actions)
        return action


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    for episode_id in range(NUM_EPISODES):
        terminal = False
        frame_id = 0
        observation = env.reset()
        while not terminal:
            action = agent.get_action(observation)
            observation, reward, terminal, _ = env.step(action)
            image = env.render(mode="rgb_array")
            if episode_id == 0:
                cv2.imwrite(f"{SAVE_DIR}/{episode_id}_{frame_id}.png", image)
            frame_id += 1


if __name__ == '__main__':
    main()
