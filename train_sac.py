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
from hl_gauss_pytorch import HLGaussLoss
from torch import optim
from tqdm import tqdm

import wandb
from networks.backbone import AE
from networks.diffusion_policy import DiffusionPolicy
from networks.sac_tanh_policy_and_q import SacQ, SacTanhPolicy
from replay_buffer import ReplayBuffer
from utils import concat_images
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer_size", type=int, default=int(8e4))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_starts", type=int, default=2000)
    parser.add_argument("--render", type=strtobool, default="True")
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--fixed_alpha", type=float, default=None)
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--policy_model", type=str, default="tanh", choices=["tanh", "diffusion"])
    parser.add_argument("--value_dim", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="rl_practice", config=vars(args), name="SAC", save_code=True)

    # seeding
    seed = args.seed if args.seed != -1 else np.random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_SAC"
    result_dir.mkdir(parents=True, exist_ok=True)

    # save seed to file
    with open(result_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    image_dir = result_dir / "image"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_save_interval = 100
    log_step = []
    log_episode = []

    # env setup
    env = make_env(result_dir / "video")
    env.action_space.seed(seed)

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
    }[args.policy_model]
    qf1 = SacQ(
        in_channels=cnn_dim,
        action_dim=action_dim,
        hidden_dim=512,
        out_dim=args.value_dim,
        use_normalize=False,
    )
    qf2 = SacQ(
        in_channels=cnn_dim,
        action_dim=action_dim,
        hidden_dim=512,
        out_dim=args.value_dim,
        use_normalize=False,
    )
    actor = actor.to(device)
    qf1 = qf1.to(device)
    qf2 = qf2.to(device)
    lr = 1e-4
    q_optimizer = optim.AdamW(
        list(encoder.parameters()) + list(qf1.parameters()) + list(qf2.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    actor_optimizer = optim.AdamW(
        list(encoder.parameters()) + list(actor.parameters()), lr=lr, weight_decay=1e-5
    )
    if args.value_dim > 1:
        hl_gauss_loss = HLGaussLoss(
            min_value=-30,
            max_value=+30,
            num_bins=args.value_dim,
            clamp_to_range=True,
        ).to(device)

    # Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item() * 2.0
    log_alpha = torch.tensor([-4.0], requires_grad=True, device=device)
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
    global_step = 0
    score_list = []
    obs, _ = env.reset(seed=seed)
    progress_bar = tqdm(range(args.learning_starts), dynamic_ncols=True)
    curr_image_dir = None

    for episode_id in range(10000):
        if (episode_id + 1) % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{episode_id:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset()
        reward_list = []
        first_value = None

        while True:
            global_step += 1

            # select action
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
            output_enc = encoder.encode(obs_tensor).detach()
            output_dec = encoder.decode(output_enc).detach()
            if global_step < args.learning_starts:
                action = env.action_space.sample()
                progress_bar.update(1)
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
            reward /= 10.0
            rb.add(obs, next_obs, action, reward, termination or truncation)

            # render
            if args.render:
                output_dec_display = output_dec[0].detach().cpu().numpy()  # (3, 96, 96)
                observation_img = np.transpose(obs, (1, 2, 0))  # (96, 96, 3)
                reconstructed_img = np.transpose(output_dec_display, (1, 2, 0))  # (96, 96, 3)
                bgr_array = concat_images(env.render(), observation_img, reconstructed_img)
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)

            # save images for specific episodes
            if episode_id % image_save_interval == 0 and curr_image_dir is not None:
                output_dec_display = output_dec[0].detach().cpu().numpy()  # (3, 96, 96)
                observation_img = np.transpose(obs, (1, 2, 0))  # (96, 96, 3)
                reconstructed_img = np.transpose(output_dec_display, (1, 2, 0))  # (96, 96, 3)
                bgr_array = concat_images(env.render(), observation_img, reconstructed_img)
                cv2.imwrite(str(curr_image_dir / f"{global_step:08d}.png"), bgr_array)

            if termination or truncation:
                break

            obs = next_obs

            if global_step <= args.learning_starts:
                continue

            if global_step % 5 != 0:
                continue

            # training.
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                state_next = encoder.encode(data.next_observations)
                state_next = state_next.flatten(start_dim=1)
                next_state_actions, next_state_log_pi, _ = actor.get_action(state_next)
                qf1_next_target = qf1(state_next, next_state_actions)
                qf2_next_target = qf2(state_next, next_state_actions)
                if args.value_dim > 1:
                    qf1_next_target = hl_gauss_loss(qf1_next_target).unsqueeze(-1)
                    qf2_next_target = hl_gauss_loss(qf2_next_target).unsqueeze(-1)
                min_q = torch.min(qf1_next_target, qf2_next_target)
                min_qf_next_target = min_q - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            state_curr = encoder.encode(data.observations).detach()
            state_curr = state_curr.flatten(start_dim=1)
            qf1_a_values = qf1(state_curr, data.actions)
            qf2_a_values = qf2(state_curr, data.actions)
            if args.value_dim == 1:
                qf1_a_values = qf1_a_values.view(-1)
                qf2_a_values = qf2_a_values.view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            else:
                qf1_loss = hl_gauss_loss(qf1_a_values, next_q_value)
                qf2_loss = hl_gauss_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            state_curr = encoder.encode(data.observations).detach()
            state_curr = state_curr.flatten(start_dim=1)
            state_norm = state_curr.norm(dim=1)
            pi, log_pi, _ = actor.get_action(state_curr)
            qf1_pi = qf1(state_curr, pi)
            qf2_pi = qf2(state_curr, pi)
            if args.value_dim > 1:
                qf1_pi = hl_gauss_loss(qf1_pi).unsqueeze(-1)
                qf2_pi = hl_gauss_loss(qf2_pi).unsqueeze(-1)
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
                    "losses/state_norm": state_norm.mean().item(),
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

        score = info["episode"]["r"]
        score_list.append(score)
        score_list = score_list[-20:]
        recent_average_score = np.mean(score_list)

        data_dict = {
            "global_step": global_step,
            "episodic_return": info["episode"]["r"],
            "episodic_length": info["episode"]["l"],
            "recent_average_score": recent_average_score,
        }
        wandb.log(data_dict)

        log_episode.append(data_dict)
        log_episode_df = pd.DataFrame(log_episode)
        log_episode_df.to_csv(result_dir / "log_episode.tsv", sep="\t", index=False)

        is_solved = recent_average_score > env.spec.reward_threshold

        if episode_id % 5 == 0 or is_solved:
            print(
                f"Ep: {episode_id}\tStep: {global_step}\tLast score: {score:.2f}\tAverage score: {recent_average_score:.2f}\tLength: {info['episode']['l']:.2f}"
            )

        # setup image directory for next episode if needed
        if (episode_id + 1) % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{episode_id + 1:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)
        else:
            curr_image_dir = None

        episode_id += 1

        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break

    env.close()
