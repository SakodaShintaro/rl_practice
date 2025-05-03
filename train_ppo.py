# Reference: https://github.com/xtma/pytorch_car_caring
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import wandb
from networks.ppo_beta_policy_and_value import PpoBetaPolicyAndValue
from wrappers import STACK_SIZE, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


class Agent:
    """
    Agent for training
    """

    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity = 2000
    batch_size = 128

    def __init__(self) -> None:
        self.training_step = 0
        self.net = PpoBetaPolicyAndValue(STACK_SIZE * 3, 3).double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state: np.ndarray) -> tuple:
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        return self.net.get_action(state)

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

        ave_action_loss_list = []
        ave_value_loss_list = []
        for _ in range(self.ppo_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
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
                sum_action_loss += action_loss.item() * len(index)
                sum_value_loss += value_loss.item() * len(index)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)
        result_dict = {}
        for i in range(len(ave_action_loss_list)):
            result_dict[f"ppo/action_loss_{i}"] = ave_action_loss_list[i]
            result_dict[f"ppo/value_loss_{i}"] = ave_value_loss_list[i]
        return result_dict


if __name__ == "__main__":
    args = parse_args()

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_PPO"
    result_dir.mkdir(parents=True, exist_ok=True)
    video_dir = result_dir / "video"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    transition = np.dtype(
        [
            ("s", np.float64, (STACK_SIZE * 3, 96, 96)),
            ("a", np.float64, (3,)),
            ("a_logp", np.float64),
            ("r", np.float64),
            ("s_", np.float64, (STACK_SIZE * 3, 96, 96)),
        ]
    )

    agent = Agent()
    env = make_env(video_dir=video_dir)

    wandb.init(project="cleanRL", config=vars(args), name="PPO", monitor_gym=True, save_code=True)

    log_episode = []
    log_step = []
    running_score = 0
    global_step = 0
    for i_ep in range(100000):
        score = 0
        state, _ = env.reset()

        while True:
            global_step += 1
            action, a_logp = agent.select_action(state)
            state_, reward, done, die, info = env.step(
                action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
            )

            # render
            rgb_array = env.render()
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            cv2.imshow("CarRacing", bgr_array)
            cv2.waitKey(1)

            if agent.store((state, action, a_logp, reward, state_)):
                print("updating")
                data_dict = agent.update()
                data_dict["global_step"] = global_step
                wandb.log(data_dict)
                fixed_data = {k.replace("ppo/", ""): v for k, v in data_dict.items()}
                log_step.append(fixed_data)
                log_step_df = pd.DataFrame(log_step)
                log_step_df.to_csv(result_dir / "log_step.tsv", sep="\t", index=False)

            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            print(f"Ep {i_ep}\tLast score: {score:.2f}\tMoving average score: {running_score:.2f}")
            data_dict = {
                "global_step": global_step,
                "episode": i_ep,
                "score": score,
                "running_score": running_score,
                "episodic_return": info["episode"]["r"],
                "episodic_length": info["episode"]["l"],
            }
            wandb.log(data_dict)

            log_episode.append(data_dict)
            log_episode_df = pd.DataFrame(log_episode)
            log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)
        if running_score > env.spec.reward_threshold:
            print(
                f"Solved! Running reward is now {running_score} and the last episode runs to {score}!"
            )
            break
