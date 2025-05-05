# Reference: https://github.com/xtma/pytorch_car_caring
import argparse
import os
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim

import wandb
from networks.ppo_beta_policy_and_value import PpoBetaPolicyAndValue
from networks.ppo_tanh_policy_and_value import PpoTanhPolicyAndValue
from networks.sequence_compressor import SequenceCompressor
from wrappers import STACK_SIZE, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--buffer_capacity", type=int, default=2000)
    parser.add_argument("--render", type=strtobool, default="True")
    parser.add_argument("--seq_len", type=int, default=1)
    return parser.parse_args()


class SequentialBatchSampler:
    def __init__(self, buffer_capacity, batch_size, k_frames, drop_last=False):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.k_frames = k_frames
        self.drop_last = drop_last

    def __iter__(self):
        valid_starts = range(self.buffer_capacity - self.k_frames + 1)
        randomized_starts = torch.randperm(len(valid_starts)).tolist()

        batch = []
        for start_idx in randomized_starts:
            seq_indices = list(range(start_idx, start_idx + self.k_frames))
            batch.append(seq_indices)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        num_samples = self.buffer_capacity - self.k_frames + 1
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size


class Agent:
    """
    Agent for training
    """

    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    batch_size = 128
    gamma = 0.99

    def __init__(self, buffer_capacity, seq_len) -> None:
        self.buffer_capacity = buffer_capacity
        self.seq_len = seq_len
        self.training_step = 0
        network_type = "beta"
        self.net = {
            "beta": PpoBetaPolicyAndValue(STACK_SIZE * 3, 3).to(device),
            "tanh": PpoTanhPolicyAndValue(STACK_SIZE * 3, 3).to(device),
        }[network_type]
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        if self.seq_len > 1:
            self.sequential_compressor = SequenceCompressor(seq_len=self.seq_len).to(device)
            self.optimizer_sc = optim.Adam(self.sequential_compressor.parameters(), lr=1e-3)

    def select_action(self, state: np.ndarray) -> tuple:
        state = torch.from_numpy(state).to(device).unsqueeze(0)
        action, a_logp = self.net.get_action(state)
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

        s = torch.tensor(self.buffer["s"]).to(device)
        a = torch.tensor(self.buffer["a"]).to(device)
        r = torch.tensor(self.buffer["r"]).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer["s_"]).to(device)
        old_a_logp = torch.tensor(self.buffer["a_logp"]).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net.get_value(s_)
            adv = target_v - self.net.get_value(s)
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # noqa: ERA001

        ave_action_loss_list = []
        ave_value_loss_list = []
        for _ in range(self.ppo_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
            for indices in SequentialBatchSampler(
                self.buffer_capacity, self.batch_size, k_frames=self.seq_len + 1, drop_last=False
            ):
                indices = np.array(indices, dtype=np.int64)
                index = indices[:, -1]
                a_logp = self.net.calc_action_logp(s[index], a[index])
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

                # update sequence compressor
                if self.seq_len > 1:
                    out = self.sequential_compressor(
                        s[indices][:, :-1],
                        r[indices][:, :-1],
                        a[indices][:, :-1],
                    )
                    loss_sc = F.mse_loss(out, torch.ones_like(out))
                    self.optimizer_sc.zero_grad()
                    loss_sc.backward()
                    self.optimizer_sc.step()
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

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

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
            ("s", np.float32, (STACK_SIZE * 3, 96, 96)),
            ("a", np.float32, (3,)),
            ("a_logp", np.float32),
            ("r", np.float32),
            ("s_", np.float32, (STACK_SIZE * 3, 96, 96)),
        ]
    )

    agent = Agent(args.buffer_capacity, args.seq_len)
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
            if args.render:
                rgb_array = env.render()
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)

            if agent.store((state, action, a_logp, reward, state_)):
                print("updating")
                data_dict = agent.update()
                data_dict["global_step"] = global_step
                data_dict["a_logp"] = a_logp
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
            print(
                f"Ep: {i_ep}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {running_score:.2f}"
            )
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
