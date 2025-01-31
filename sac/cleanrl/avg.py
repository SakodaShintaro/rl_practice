"""Based on https://github.com/gauthamvasan/avg/blob/main/avg.py.

Copyright (c) [2024] [Gautham Vasan] - MIT License.
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import wandb
from network import Actor, SoftQNetwork
from reward_processor import RewardProcessor


class AVG:
    """AVG Agent."""

    def __init__(self, cfg: argparse.Namespace, env: gym.Env) -> None:
        self.steps = 0

        self.actor = Actor(env)
        self.Q = SoftQNetwork(env)

        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr

        self.popt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.betas,
            weight_decay=cfg.l2_actor,
        )
        self.qopt = torch.optim.AdamW(
            self.Q.parameters(),
            lr=cfg.critic_lr,
            betas=cfg.betas,
            weight_decay=cfg.l2_critic,
        )

        self.alpha_lr, self.gamma, self.device = cfg.alpha_lr, cfg.gamma, cfg.device

        self.use_eligibility_trace = cfg.use_eligibility_trace

        self.et_lambda = cfg.et_lambda
        with torch.no_grad():
            self.eligibility_traces_q = [
                torch.zeros_like(p, requires_grad=False) for p in self.Q.parameters()
            ]

    def compute_action(self, obs: np.ndarray) -> tuple[torch.Tensor, dict]:
        """Compute the action and action information given an observation."""
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, log_prob, mean = self.actor.get_action(obs)
        return action, log_prob, mean

    def update(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        lprob: torch.Tensor,
    ) -> None:
        """Update the actor and critic networks based on the observed transition."""
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs = torch.Tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, lprob = action.to(self.device), lprob.to(self.device)

        #### Q loss
        q = self.Q(obs, action.detach())  # N.B: Gradient should NOT pass through action here
        with torch.no_grad():
            next_action, next_lprob, mean = self.actor.get_action(next_obs)
            q2 = self.Q(next_obs, next_action)
            target_V = q2 - self.alpha_lr * next_lprob

        delta = reward + (1 - done) * self.gamma * target_V - q
        ####

        # Policy loss
        ploss = self.alpha_lr * lprob - self.Q(obs, action)  # N.B: USE reparametrized action
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        if self.use_eligibility_trace:
            q.backward()
            with torch.no_grad():
                for p, et in zip(self.Q.parameters(), self.eligibility_traces_q):
                    et.mul_(self.et_lambda * self.gamma).add_(p.grad.data)
                    p.grad.data = -2.0 * delta * et
        else:
            qloss = delta**2
            qloss.backward()
        self.qopt.step()

        self.steps += 1

        return {
            "delta": delta.item(),
            "q": q.item(),
        }

    def save(self, model_dir: str, unique_str: str) -> None:
        """Save the model parameters to a file."""
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
        }
        torch.save(model, f"{model_dir}/{unique_str}.pt")

    def reset_eligibility_traces(self) -> None:
        """Reset eligibility traces."""
        for et in self.eligibility_traces_q:
            et.zero_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v5", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generator")
    parser.add_argument("--N", default=2_000_000, type=int, help="# total timesteps for the run")
    # SAVG params
    parser.add_argument("--actor_lr", default=0.0063, type=float, help="Actor step size")
    parser.add_argument("--critic_lr", default=0.0087, type=float, help="Critic step size")
    parser.add_argument("--beta1", default=0.0, type=float, help="Beta1 parameter of Adam")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--alpha_lr", default=0.07, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument("--l2_actor", default=0.0, type=float, help="L2 Regularization")
    parser.add_argument("--l2_critic", default=0.0, type=float, help="L2 Regularization")
    parser.add_argument("--nhid_actor", default=256, type=int)
    parser.add_argument("--nhid_critic", default=256, type=int)
    parser.add_argument("--use_eligibility_trace", action="store_true")
    parser.add_argument("--et_lambda", default=0.0, type=float)
    parser.add_argument("--reward_processing_type", default="none", type=str)
    parser.add_argument("--additional_coeff", default=2.5, type=float)
    # Miscellaneous
    parser.add_argument("--checkpoint", default=1_000_000, type=int, help="Checkpoint interval")
    parser.add_argument("--save_dir", default="./results", type=Path, help="Location to store")
    parser.add_argument("--save_suffix", default="avg", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--n_eval", default=0, type=int, help="Number of eval episodes")
    parser.add_argument("--print_interval_episode", default=50, type=int)
    parser.add_argument("--record_interval_episode", default=2000, type=int)
    args = parser.parse_args()

    # init wandb
    wandb.init(project="cleanRL", name=args.save_suffix, config=args)

    # Adam
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = args.save_dir / f"{datetime_str}_{args.save_suffix}"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Start experiment
    # N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow.
    # Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    tic = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{save_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"AVG-{args.env}_seed-{args.seed}")

    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Env
    env = gym.make(args.env, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)

    #### Reproducibility
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Learner
    logger.info(f"{env.observation_space=}")
    logger.info(f"{env.action_space=}")
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    agent = AVG(args, env)

    action_coeff = (env.action_space.high - env.action_space.low) / 2

    # なぜか追加の係数がないとHumanoid-v5で学習が進まない
    logger.info(f"Before {action_coeff=}")
    action_coeff *= args.additional_coeff
    logger.info(f"After  {action_coeff=}")

    # Interaction
    reward_processor = RewardProcessor(args.reward_processing_type)
    episode_stats = defaultdict(list)
    episode_id = 1
    sum_reward, ep_step = 0, 0
    sum_delta, sum_lprob = 0, 0
    sum_reward_normed = 0
    terminated, truncated = False, False
    obs, _ = env.reset()
    data_list = []

    for total_step in range(1, args.N + 1):
        # N.B: Action is a torch.Tensor
        action, lprob, mean = agent.compute_action(obs)
        sim_action = action.detach().cpu().view(-1).numpy()
        sim_action *= action_coeff

        # Receive reward and next state
        next_obs, reward, terminated, truncated, _ = env.step(sim_action)
        reward_normed = reward_processor.normalize(reward)
        if episode_id % args.record_interval_episode == 0:
            save_image_dir = save_dir / f"images/{episode_id:06d}"
            save_image_dir.mkdir(exist_ok=True, parents=True)
            image = env.render()
            cv2.imwrite(str(save_image_dir / f"{ep_step:08d}.png"), image)
        stats = agent.update(obs, action, next_obs, reward_normed, terminated, lprob)
        sum_delta += stats["delta"]
        sum_lprob += lprob.item()
        sum_reward += reward
        sum_reward_normed += reward_normed
        ep_step += 1

        obs = next_obs

        if total_step % args.checkpoint == 0:
            agent.save(
                model_dir=save_dir,
                unique_str=f"model_{total_step:010d}",
            )

        # Termination
        if terminated or truncated:
            curr_ave_delta = sum_delta / ep_step
            curr_ave_lprob = sum_lprob / ep_step
            duration_total_sec = int(time.time() - tic)

            data_dict = {
                "charts/elapse_time_sec": duration_total_sec,
                "episode_id": episode_id,
                "episodic_length": ep_step,
                "episodic_return": sum_reward,
                "episodic_return_normed": sum_reward_normed,
                "losses/qf1_loss": curr_ave_delta,
                "losses/log_pi": curr_ave_lprob,
                "global_step": total_step,
            }

            data_list.append(data_dict)
            df = pd.DataFrame(data_list)
            df.to_csv(f"{save_dir}/result.tsv", index=False, sep="\t")
            wandb.log(data_dict)

            episode_stats["episode_id"].append(episode_id)
            episode_stats["steps"].append(ep_step)
            episode_stats["return"].append(sum_reward)
            episode_stats["return_normed"].append(sum_reward_normed)
            episode_stats["ave_delta"].append(curr_ave_delta)
            episode_stats["ave_lprob"].append(curr_ave_lprob)

            if episode_id % args.print_interval_episode == 0:
                ave_return = np.mean(episode_stats["return"])
                ave_return_normed = np.mean(episode_stats["return_normed"])
                ave_steps = np.mean(episode_stats["steps"])
                ave_delta = np.mean(episode_stats["ave_delta"])
                ave_lprob = np.mean(episode_stats["ave_lprob"])
                episode_stats = defaultdict(list)
                duration_min = duration_total_sec // 60
                duration_hor = duration_min // 60
                duration_sec = duration_total_sec % 60
                duration_min = duration_min % 60
                duration_str = f"{duration_hor:03d}h:{duration_min:02d}m:{duration_sec:02d}s"
                logger.info(
                    f"{duration_str}\t"
                    f"Episode: {episode_id:,}\t"
                    f"Step: {ave_steps:7.2f}\t"
                    f"Return: {ave_return:.2f}\t"
                    f"ReturnNormed: {ave_return_normed:.2f}\t"
                    f"Delta: {ave_delta:.2f}\t"
                    f"lprob: {ave_lprob:.2f}\t"
                    f"TotalStep: {total_step:,}",
                )

            obs, _ = env.reset()
            agent.reset_eligibility_traces()
            sum_reward, ep_step = 0, 0
            sum_delta, sum_q = 0, 0
            sum_reward_normed = 0
            episode_id += 1

    # Save returns and args before exiting run
    agent.save(model_dir=save_dir, unique_str="last_model")

    df = pd.DataFrame(data_list)
    df.to_csv(f"{save_dir}/result.tsv", index=False, sep="\t")
