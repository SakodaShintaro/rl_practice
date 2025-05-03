# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

import wandb
from networks.backbone import BaseCNN
from networks.network import SacQ, SacTanhPolicy
from replay_buffer import ReplayBuffer
from wrappers import STACK_SIZE, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer_size", type=int, default=int(1e5))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_starts", type=int, default=1e3)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    wandb.init(project="cleanRL", config=vars(args), name="SAC", monitor_gym=True, save_code=True)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_SAC"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_step = []
    log_episode = []

    # env setup
    env = make_env(result_dir / "video")
    env.action_space.seed(args.seed)

    action_scale = (env.action_space.high - env.action_space.low) / 2.0
    action_bias = (env.action_space.high + env.action_space.low) / 2.0

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    action_dim = np.prod(env.action_space.shape)
    encoder = BaseCNN(in_channels=3 * STACK_SIZE).to(device)
    actor = SacTanhPolicy(
        in_channels=3 * STACK_SIZE, action_dim=action_dim, hidden_dim=256, use_normalize=False
    ).to(device)
    qf1 = SacQ(
        in_channels=3 * STACK_SIZE, action_dim=action_dim, hidden_dim=256, use_normalize=False
    ).to(device)
    qf2 = SacQ(
        in_channels=3 * STACK_SIZE, action_dim=action_dim, hidden_dim=256, use_normalize=False
    ).to(device)
    q_optimizer = optim.Adam(
        list(encoder.parameters()) + list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(
        list(encoder.parameters()) + list(actor.parameters()), lr=args.policy_lr
    )

    # Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space.shape,
        env.action_space.shape,
        device,
    )
    start_time = time.time()

    # start the game
    obs, _ = env.reset(seed=args.seed)
    progress_bar = tqdm(range(args.total_timesteps), dynamic_ncols=True)
    for global_step in range(args.total_timesteps):
        # put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = actor.get_action(torch.Tensor(obs).to(device).unsqueeze(0))
            action = action[0].detach().cpu().numpy()
            action = action * action_scale + action_bias

        # execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        rb.add(obs, next_obs, action, reward, termination or truncation)

        # render
        rgb_array = env.render()
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("CarRacing", bgr_array)
        cv2.waitKey(1)

        if termination or truncation:
            data_dict = {
                "global_step": global_step,
                "episodic_return": info["episode"]["r"],
                "episodic_length": info["episode"]["l"],
            }
            wandb.log(data_dict)

            log_episode.append(data_dict)
            log_episode_df = pd.DataFrame(log_episode)
            log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)

            obs, _ = env.reset()
        else:
            obs = next_obs

        progress_bar.update(1)

        if global_step <= args.learning_starts:
            continue

        # training.
        data = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
            qf1_next_target = qf1(data.next_observations, next_state_actions)
            qf2_next_target = qf2(data.next_observations, next_state_actions)
            min_q = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_q - alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                min_qf_next_target
            ).view(-1)

        qf1_a_values = qf1(data.observations, data.actions).view(-1)
        qf2_a_values = qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        pi, log_pi, _ = actor.get_action(data.observations)
        qf1_pi = qf1(data.observations, pi)
        qf2_pi = qf2(data.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        with torch.no_grad():
            _, log_pi, _ = actor.get_action(data.observations)
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        alpha = log_alpha.exp().item()

        if global_step % 100 == 0:
            elapsed_time = time.time() - start_time
            data_dict = {
                "global_step": global_step,
                "losses/qf1_values": qf1_a_values.mean().item(),
                "losses/qf2_values": qf2_a_values.mean().item(),
                "losses/qf1_loss": qf1_loss.item(),
                "losses/qf2_loss": qf2_loss.item(),
                "losses/qf_loss": qf_loss.item() / 2.0,
                "losses/actor_loss": actor_loss.item(),
                "losses/alpha": alpha,
                "losses/log_pi": log_pi.mean().item(),
                "losses/alpha_loss": alpha_loss.item(),
                "charts/elapse_time_sec": elapsed_time,
                "charts/SPS": int(global_step / elapsed_time),
                "reward": reward,
            }
            wandb.log(data_dict)

            fixed_data = {
                k.replace("losses/", "").replace("charts/", ""): v for k, v in data_dict.items()
            }
            log_step.append(fixed_data)
            log_step_df = pd.DataFrame(log_step)
            log_step_df.to_csv(
                result_dir / "log_step.tsv", sep="\t", index=False, float_format="%.3f"
            )

    env.close()
