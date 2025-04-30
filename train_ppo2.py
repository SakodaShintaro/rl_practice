# Reference) https://github.com/xtma/pytorch_car_caring
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--img-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat the same action for multiple steps
    """

    def __init__(self, env, repeat=8):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0
        for i in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class AverageRewardEarlyStopWrapper(gym.Wrapper):
    """
    End episode early if average reward over last some steps is too low
    """

    def __init__(self, env):
        super().__init__(env)
        self.window_size = 50
        self.threshold = -0.1
        self.recent_rewards = []

    def reset(self, **kwargs):
        self.recent_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.recent_rewards.append(reward)
        self.recent_rewards = self.recent_rewards[-self.window_size :]

        if len(self.recent_rewards) >= self.window_size:
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward <= self.threshold:
                terminated = True

        return obs, reward, terminated, truncated, info


class DieStateRewardWrapper(gym.Wrapper):
    """
    Don't penalize "die state" and add bonus reward if terminated
    """

    def __init__(self, env, bonus_reward=100):
        super().__init__(env)
        self.bonus_reward = bonus_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            reward += self.bonus_reward

        return obs, reward, terminated, truncated, info


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn_base = nn.Sequential(  # input shape (args.img_stack * 3, 96, 96)
            nn.Conv2d(args.img_stack * 3, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m: object) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x: torch.Tensor) -> tuple:
        # x.shape = (batch_size, args.img_stack, 96, 96, 3)
        bs, st, h, w, c = x.shape
        x = x.permute((0, 1, 4, 2, 3))  # (batch_size, args.img_stack, 3, 96, 96)
        x = x.reshape(bs, st * c, h, w)
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent:
    """
    Agent for training
    """

    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self) -> None:
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state: np.ndarray) -> tuple:
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def store(self, transition: tuple) -> bool:
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self) -> None:
        self.training_step += 1

        s = torch.tensor(self.buffer["s"], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer["a"], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer["r"], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer["s_"], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer["a_logp"], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # noqa: ERA001

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, drop_last=False
            ):
                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                )
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2.0 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    args = parse_args()

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_PPO"
    result_dir.mkdir(parents=True, exist_ok=True)
    video_dir = result_dir / "video"
    log_episode = []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    transition = np.dtype(
        [
            ("s", np.float64, (args.img_stack, 96, 96, 3)),
            ("a", np.float64, (3,)),
            ("a_logp", np.float64),
            ("r", np.float64),
            ("s_", np.float64, (args.img_stack, 96, 96, 3)),
        ]
    )

    agent = Agent()
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = ActionRepeatWrapper(env, repeat=args.action_repeat)
    env = AverageRewardEarlyStopWrapper(env)
    env = DieStateRewardWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(
        env, video_folder=video_dir, episode_trigger=lambda x: x % 200 == 0
    )

    training_records = []
    running_score = 0
    for i_ep in range(100000):
        score = 0
        state, _ = env.reset()

        while True:
            action, a_logp = agent.select_action(state)
            state_, reward, done, die, _ = env.step(
                action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
            )
            if args.render:
                rgb_array = env.render()
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)
            if agent.store((state, action, a_logp, reward, state_)):
                print("updating")
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            print(f"Ep {i_ep}\tLast score: {score:.2f}\tMoving average score: {running_score:.2f}")
            data_dict = {
                "episode": i_ep,
                "score": score,
                "running_score": running_score,
            }
            log_episode.append(data_dict)
            log_episode_df = pd.DataFrame(log_episode)
            log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)
        if running_score > env.spec.reward_threshold:
            print(
                f"Solved! Running reward is now {running_score} and the last episode runs to {score}!"
            )
            break
