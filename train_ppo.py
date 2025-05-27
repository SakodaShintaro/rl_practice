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
from networks.ppo_paligemma_policy_value import PpoPaligemmaPolicyAndValue
from utils import concat_images
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--buffer_capacity", type=int, default=2000)
    parser.add_argument("--render", type=strtobool, default="True")
    parser.add_argument("--seq_len", type=int, default=2)
    parser.add_argument(
        "--model_name", type=str, default="default", choices=["default", "paligemma"]
    )
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
    clip_param_policy = 0.1
    clip_param_value = 1.0
    ppo_epoch = 10
    batch_size = 128
    gamma = 0.99

    def __init__(self, buffer_capacity, seq_len, model_name) -> None:
        self.buffer_capacity = buffer_capacity
        self.seq_len = seq_len
        self.training_step = 0
        self.net = {
            "default": PpoBetaPolicyAndValue(3, seq_len).to(device),
            "paligemma": PpoPaligemmaPolicyAndValue(3).to(device),
        }[model_name]
        self.buffer = np.empty(
            self.buffer_capacity,
            dtype=np.dtype(
                [
                    ("s", np.float32, (3, 96, 96)),
                    ("a", np.float32, (3,)),
                    ("a_logp", np.float32),
                    ("r", np.float32),
                    ("v", np.float32),
                    ("done", np.int32),
                ]
            ),
        )
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.r_list = []
        self.s_list = []
        self.a_list = []

    @torch.inference_mode()
    def select_action(self, reward: float, state: np.ndarray) -> tuple:
        reward = torch.from_numpy(np.array(reward)).to(device).unsqueeze(0)
        state = torch.from_numpy(state).to(device).unsqueeze(0)
        self.r_list.append(reward)
        self.r_list = self.r_list[-self.seq_len :]
        self.s_list.append(state)
        self.s_list = self.s_list[-self.seq_len :]

        curr_r = torch.cat(self.r_list, dim=0).unsqueeze(0).unsqueeze(-1)
        curr_s = torch.cat(self.s_list, dim=0).unsqueeze(0)
        a_with_dummy = self.a_list + [torch.tensor([[0.0, 0.0, 0.0]], device=device)]
        curr_a = torch.cat(a_with_dummy, dim=0).unsqueeze(0)

        if curr_r.shape[1] < self.seq_len:
            padding_size = self.seq_len - curr_r.shape[1]
            pad_r = torch.zeros(1, padding_size, *curr_r.shape[2:], device=device)
            pad_s = torch.zeros(1, padding_size, *curr_s.shape[2:], device=device)
            pad_a = torch.zeros(1, padding_size, *curr_a.shape[2:], device=device)
            curr_r = torch.cat((pad_r, curr_r), dim=1)
            curr_s = torch.cat((pad_s, curr_s), dim=1)
            curr_a = torch.cat((pad_a, curr_a), dim=1)
        assert curr_r.shape[1] == self.seq_len
        assert curr_s.shape[1] == self.seq_len
        assert curr_a.shape[1] == self.seq_len

        result_dict = self.net(curr_r, curr_s, curr_a)
        action = result_dict["action"]
        a_logp = result_dict["a_logp"]
        value = result_dict["value"]
        self.a_list.append(action)
        self.a_list = self.a_list[-self.seq_len :]
        if len(self.a_list) == self.seq_len:
            self.a_list = self.a_list[1:]

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        value = value.item()
        return action, a_logp, value, result_dict

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
        v = torch.tensor(self.buffer["v"]).to(device).view(-1, 1)
        old_a_logp = torch.tensor(self.buffer["a_logp"]).to(device).view(-1, 1)
        done = torch.tensor(self.buffer["done"]).to(device).view(-1, 1)

        target_v = r[:-1] + (1 - done[:-1]) * self.gamma * v[1:]
        adv = target_v - v[:-1]
        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # noqa: ERA001

        ave_action_loss_list = []
        ave_value_loss_list = []
        ave_pred_s_loss_list = []
        for _ in range(self.ppo_epoch):
            sum_action_loss = 0.0
            sum_value_loss = 0.0
            sum_pred_s_loss = 0.0
            for indices in SequentialBatchSampler(
                self.buffer_capacity - 1,
                self.batch_size,
                k_frames=self.seq_len,
                drop_last=False,
            ):
                indices = np.array(indices, dtype=np.int64)
                index = indices[:, -1]
                curr_action = a[indices][:, :-1]
                dummy_action = torch.zeros((curr_action.shape[0], 1, 3), device=device)
                curr_action = torch.cat((curr_action, dummy_action), dim=1)

                net_out_dict = self.net(r[indices], s[indices], curr_action, a[index])
                a_logp = net_out_dict["a_logp"]
                value = net_out_dict["value"]
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy)
                    * adv[index]
                )
                action_loss = -torch.min(surr1, surr2).mean()

                value_clipped = torch.clamp(
                    value, v[index] - self.clip_param_value, v[index] + self.clip_param_value
                )
                value_loss_unclipped = F.smooth_l1_loss(value, target_v[index])
                value_loss_clipped = F.smooth_l1_loss(value_clipped, target_v[index])
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

                pred_error = net_out_dict["error"]
                pred_error_s = pred_error[:, 3::3]  # 先頭は明らかに予測不可能なので3から
                pred_error_a = pred_error[:, 1::3]
                pred_error_r = pred_error[:, 2::3]
                pred_loss_s = pred_error_s.mean()

                loss = action_loss + 2.0 * value_loss + pred_loss_s
                sum_action_loss += action_loss.item() * len(index)
                sum_value_loss += value_loss.item() * len(index)
                sum_pred_s_loss += pred_loss_s.item() * len(index)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

            ave_action_loss = sum_action_loss / self.buffer_capacity
            ave_value_loss = sum_value_loss / self.buffer_capacity
            ave_pred_s_loss = sum_pred_s_loss / self.buffer_capacity
            ave_action_loss_list.append(ave_action_loss)
            ave_value_loss_list.append(ave_value_loss)
            ave_pred_s_loss_list.append(ave_pred_s_loss)
        result_dict = {}
        ratio_list = []
        result_dict["ppo/average_action_loss"] = np.mean(ave_action_loss_list)
        result_dict["ppo/average_value_loss"] = np.mean(ave_value_loss_list)
        result_dict["ppo/average_pred_s_loss"] = np.mean(ave_pred_s_loss_list)
        result_dict["ppo/average_ratio"] = np.mean(ratio_list)
        result_dict["ppo/average_target_v"] = target_v.mean().item()
        result_dict["ppo/average_adv"] = adv.mean().item()
        return result_dict


if __name__ == "__main__":
    args = parse_args()

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_PPO"
    result_dir.mkdir(parents=True, exist_ok=True)
    video_dir = result_dir / "video"
    image_dir = result_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_save_interval = 100

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    with open(result_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    agent = Agent(args.buffer_capacity, args.seq_len, args.model_name)
    env = make_env(video_dir=video_dir)

    wandb.init(project="rl_practice", config=vars(args), name="PPO", save_code=True)

    log_episode = []
    log_step = []
    score_list = []
    global_step = 0
    reward = 0.0
    for i_ep in range(100000):
        state, _ = env.reset()

        if i_ep % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{i_ep:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)

        reward_list = []
        first_value = None

        while True:
            global_step += 1
            action, a_logp, value, net_out_dict = agent.select_action(reward, state)
            state_, reward, done, die, info = env.step(
                action * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])
            )
            done = bool(done or die)
            normed_reward = reward / 10.0
            if len(reward_list) == 0:
                first_value = value
            reward_list.append(reward)

            # render
            ae = agent.net.sequential_processor.state_encoder
            predicted_s = net_out_dict["predicted_s"]
            predicted_s = predicted_s.view(1, 4, 12, 12)
            output_dec = ae.decode(predicted_s).detach().cpu().numpy()[0]
            observation_img = np.transpose(state, (1, 2, 0))  # (96, 96, 3)
            reconstructed_img = np.transpose(output_dec, (1, 2, 0))  # (96, 96, 3)
            bgr_array = concat_images(env.render(), observation_img, reconstructed_img)
            if args.render:
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)
            if i_ep % image_save_interval == 0:
                cv2.imwrite(str(curr_image_dir / f"{global_step:08d}.png"), bgr_array)

            data_dict = {
                "global_step": global_step,
                "a_logp": a_logp,
                "value": value,
                "reward": reward,
                "normed_reward": normed_reward,
            }

            for key in ["x", "value_x", "policy_x"]:
                value_tensor = net_out_dict[key]
                data_dict[f"activation/{key}_norm"] = value_tensor.norm(dim=1).mean().item()
                data_dict[f"activation/{key}_mean"] = value_tensor.mean(dim=1).mean().item()
                data_dict[f"activation/{key}_std"] = value_tensor.std(dim=1).mean().item()

            if agent.store((state, action, a_logp, normed_reward, value, done)):
                print("updating", end="\r")
                train_result = agent.update()
                data_dict.update(train_result)
                fixed_data = {k.replace("ppo/", ""): v for k, v in data_dict.items()}
                log_step.append(fixed_data)
                log_step_df = pd.DataFrame(log_step)
                log_step_df.to_csv(result_dir / "log_step.tsv", sep="\t", index=False)

                for name, p in agent.net.named_parameters():
                    data_dict[f"params/{name}"] = p.norm().item()

                wandb.log(data_dict)
            elif global_step % 100 == 0:
                wandb.log(data_dict)

            state = state_
            if done or die:
                break
        score = info["episode"]["r"]
        score_list.append(score)
        score_list = score_list[-20:]
        recent_average_score = np.mean(score_list)
        is_solved = recent_average_score > env.spec.reward_threshold

        weighted_reward = 0.0
        coeff = 1.0
        for r in reward_list:
            weighted_reward += coeff * r
            coeff *= agent.gamma

        if i_ep % args.log_interval == 0 or is_solved:
            print(
                f"Ep: {i_ep}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {info['episode']['l']:.2f}"
            )
        data_dict = {
            "global_step": global_step,
            "episode": i_ep,
            "score": score,
            "recent_average_score": recent_average_score,
            "episodic_return": info["episode"]["r"],
            "episodic_length": info["episode"]["l"],
            "weighted_reward": weighted_reward,
            "first_value": first_value,
        }
        wandb.log(data_dict)

        log_episode.append(data_dict)
        log_episode_df = pd.DataFrame(log_episode)
        log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)
        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break
