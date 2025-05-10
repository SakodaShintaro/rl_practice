# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from datetime import datetime
from distutils.util import strtobool
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
from networks.backbone import AE
from networks.diffusion_policy import DiffusionPolicy
from networks.sac_tanh_policy_and_q import SacQ, SacTanhPolicy
from replay_buffer import ReplayBuffer
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer_size", type=int, default=int(8e4))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_starts", type=int, default=4000)
    parser.add_argument("--render", type=strtobool, default="True")
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--fixed_alpha", type=float, default=None)
    parser.add_argument("--action_noise", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

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

    action_low = env.action_space.low
    action_high = env.action_space.high
    print(f"action_low: {action_low}, action_high: {action_high}")
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    print(f"action_scale: {action_scale}, action_bias: {action_bias}")

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    action_dim = np.prod(env.action_space.shape)
    encoder = AE().to(device)
    cnn_dim = 576
    actor = {
        "tanh": SacTanhPolicy(
            in_channels=cnn_dim, action_dim=action_dim, hidden_dim=512, use_normalize=False
        ),
        "diffusion": DiffusionPolicy(state_dim=cnn_dim, action_dim=action_dim, use_normalize=False),
    }["tanh"]
    qf1 = SacQ(in_channels=cnn_dim, action_dim=action_dim, hidden_dim=512, use_normalize=False)
    qf2 = SacQ(in_channels=cnn_dim, action_dim=action_dim, hidden_dim=512, use_normalize=False)
    actor = actor.to(device)
    qf1 = qf1.to(device)
    qf2 = qf2.to(device)
    lr = 3e-4
    q_optimizer = optim.AdamW(
        list(encoder.parameters()) + list(qf1.parameters()) + list(qf2.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    actor_optimizer = optim.AdamW(
        list(encoder.parameters()) + list(actor.parameters()), lr=lr, weight_decay=1e-5
    )

    # Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item() * 2.0
    log_alpha = torch.tensor([-2.0], requires_grad=True, device=device)
    alpha = log_alpha.exp().item() if args.fixed_alpha is None else args.fixed_alpha
    a_optimizer = optim.Adam([log_alpha], lr=lr)
    print(f"{target_entropy=}")

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
        obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
        output_enc = encoder.encode(obs_tensor).detach()
        output_dec = encoder.decode(output_enc).detach()
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            output_enc = output_enc.flatten(start_dim=1)
            action, selected_log_pi, _ = actor.get_action(output_enc)
            action = action[0].detach().cpu().numpy()
            action = action * action_scale + action_bias

            action_noise = env.action_space.sample()
            c = args.action_noise
            action = (1 - c) * action + c * action_noise
            action = np.clip(action, action_low, action_high)

        # execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        reward = np.clip(reward, -3.0, 3.0)
        rb.add(obs, next_obs, action, reward, termination or truncation)

        # render
        if args.render:
            output_dec = output_dec[0].detach().cpu().numpy()  # (3, 96, 96)
            obs_to_render = np.transpose(obs, (1, 2, 0))  # (96, 96, 3)
            dec_to_render = np.transpose(output_dec, (1, 2, 0))  # (96, 96, 3)
            concat = cv2.vconcat([obs_to_render, dec_to_render])  # (192, 96, 3)
            rgb_array = env.render()  # (400, 600, 3)
            rem = rgb_array.shape[0] - concat.shape[0]
            concat = cv2.copyMakeBorder(concat, 0, rem, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            concat *= 255
            concat = np.clip(concat, 0, 255).astype(np.uint8)
            concat = cv2.hconcat([rgb_array, concat])  # (400, 696, 3)
            bgr_array = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
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

        if global_step % 20 != 0:
            continue

        # training.
        data = rb.sample(args.batch_size)
        with torch.no_grad():
            state_next = encoder.encode(data.next_observations)
            state_next = state_next.flatten(start_dim=1)
            next_state_actions, next_state_log_pi, _ = actor.get_action(state_next)
            qf1_next_target = qf1(state_next, next_state_actions)
            qf2_next_target = qf2(state_next, next_state_actions)
            min_q = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = min_q - alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                min_qf_next_target
            ).view(-1)

        state_curr = encoder.encode(data.observations).detach()
        state_curr = state_curr.flatten(start_dim=1)
        qf1_a_values = qf1(state_curr, data.actions).view(-1)
        qf2_a_values = qf2(state_curr, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        state_curr = encoder.encode(data.observations).detach()
        state_curr = state_curr.flatten(start_dim=1)
        pi, log_pi, _ = actor.get_action(state_curr)
        qf1_pi = qf1(state_curr, pi)
        qf2_pi = qf2(state_curr, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = (-log_alpha.exp() * (log_pi.detach() + target_entropy)).mean()

        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        alpha = log_alpha.exp().item() if args.fixed_alpha is None else args.fixed_alpha

        if global_step % 10 == 0:
            elapsed_time = time.time() - start_time
            data_dict = {
                "global_step": global_step,
                "losses/qf1_values": qf1_a_values.mean().item(),
                "losses/qf2_values": qf2_a_values.mean().item(),
                "losses/qf1_loss": qf1_loss.item(),
                "losses/qf2_loss": qf2_loss.item(),
                "losses/qf_loss": qf_loss.item() / 2.0,
                "losses/min_qf_next_target": min_qf_next_target.mean().item(),
                "losses/next_q_value": next_q_value.mean().item(),
                "losses/actor_loss": actor_loss.item(),
                "losses/alpha": alpha,
                "losses/log_pi": log_pi.mean().item(),
                "a_logp": selected_log_pi.mean().item(),
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
