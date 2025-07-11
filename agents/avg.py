"""Based on https://github.com/gauthamvasan/avg/blob/main/avg.py.

Copyright (c) [2024] [Gautham Vasan] - MIT License.
"""

import argparse
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from hl_gauss_pytorch import HLGaussLoss

import wandb
from networks.backbone import AE, SmolVLABackbone
from networks.sac_tanh_policy_and_q import SacQ, SacTanhPolicy
from reward_processor import RewardProcessor
from td_error_scaler import TDErrorScaler
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--image_encoder", type=str, default="ae", choices=["ae", "smolvla"])
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--actor_block_num", type=int, default=1)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--critic_block_num", type=int, default=1)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--N", default=2_000_000, type=int)
    parser.add_argument("--actor_lr", default=0.0063, type=float)
    parser.add_argument("--critic_lr", default=0.0087, type=float)
    parser.add_argument("--alpha_lr", default=1e-2, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--use_eligibility_trace", action="store_true")
    parser.add_argument("--et_lambda", default=0.0, type=float)
    parser.add_argument("--reward_processing_type", default="none", type=str)
    parser.add_argument("--print_interval_episode", default=1, type=int)
    parser.add_argument("--record_interval_episode", default=10, type=int)
    parser.add_argument("--without_entropy_term", action="store_true")
    return parser.parse_args()


class AVG:
    """AVG Agent."""

    def __init__(self, args: argparse.Namespace, env: gym.Env, device: torch.device) -> None:
        self.steps = 0

        if args.image_encoder == "ae":
            self.encoder_image = AE()
        elif args.image_encoder == "smolvla":
            self.encoder_image = SmolVLABackbone()
        else:
            raise ValueError(f"Unknown image encoder: {args.image_encoder}")
        self.encoder_image.to(device)
        self.cnn_dim = 4 * 12 * 12  # 576

        action_dim = np.prod(env.action_space.shape)
        self.actor = SacTanhPolicy(
            in_channels=self.cnn_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
        ).to(device)
        num_bins = 51
        self.Q = SacQ(
            in_channels=self.cnn_dim,
            block_num=args.critic_block_num,
            num_bins=num_bins,
            sparsity=args.sparsity,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
        ).to(device)

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr

        betas = [0.0, 0.999]
        weight_decay = 1e-5
        self.popt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.qopt = torch.optim.AdamW(
            self.Q.parameters(),
            lr=args.critic_lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.device = device
        self.gamma = args.gamma
        self.td_error_scaler = TDErrorScaler()
        self.G = 0

        self.use_eligibility_trace = args.use_eligibility_trace

        self.et_lambda = args.et_lambda
        with torch.no_grad():
            self.eligibility_traces_q = [
                torch.zeros_like(p, requires_grad=False) for p in self.Q.parameters()
            ]

        if args.without_entropy_term:
            self.log_alpha = None
        else:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
            self.log_alpha = torch.nn.Parameter(
                torch.zeros(1, requires_grad=True, device=device)
            )
            self.aopt = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)

        self.hl_gauss_loss = HLGaussLoss(
            min_value=-30,
            max_value=+30,
            num_bins=num_bins,
            clamp_to_range=True,
        ).to(device)

    def compute_action(self, obs: np.ndarray) -> tuple[torch.Tensor, dict]:
        """Compute the action and action information given an observation."""
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        obs = self.encoder_image.encode(obs)
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
        obs = self.encoder_image.encode(obs)
        next_obs = self.encoder_image.encode(next_obs)

        #### Q loss
        with torch.no_grad():
            alpha = self.log_alpha.exp().item() if self.log_alpha is not None else 0.0
            next_action, next_lprob, mean = self.actor.get_action(next_obs)
            next_q_dict = self.Q(next_obs, next_action)
            next_q_logit = next_q_dict["output"]
            next_q = self.hl_gauss_loss(next_q_logit).unsqueeze(-1)
            target_V = next_q - alpha * next_lprob

        #### Return scaling
        r_ent = reward - alpha * lprob.detach().item()
        self.G += r_ent
        if done:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.gamma, G=None)

        curr_q_dict = self.Q(obs, action.detach())
        curr_q_logit = curr_q_dict["output"]
        curr_q = self.hl_gauss_loss(curr_q_logit).unsqueeze(-1)
        delta = reward + (1 - done) * self.gamma * target_V - curr_q
        delta /= self.td_error_scaler.sigma

        # Policy loss
        curr_q_dict = self.Q(obs, action)
        curr_q_logit = curr_q_dict["output"]
        curr_q = self.hl_gauss_loss(curr_q_logit).unsqueeze(-1)
        ploss = alpha * lprob - curr_q
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        if self.use_eligibility_trace:
            curr_q_dict.backward()
            with torch.no_grad():
                for p, et in zip(self.Q.parameters(), self.eligibility_traces_q):
                    et.mul_(self.et_lambda * self.gamma).add_(p.grad.data)
                    p.grad.data = -2.0 * delta.item() * et
        else:
            qloss = delta**2
            qloss.backward()
        self.qopt.step()

        # alpha
        if self.log_alpha is None:
            alpha_loss = torch.Tensor([0.0])
        else:
            alpha_loss = (-self.log_alpha.exp() * (lprob.detach() + self.target_entropy)).mean()
            self.aopt.zero_grad()
            alpha_loss.backward()
            self.aopt.step()

        self.steps += 1

        return {
            "delta": delta.item(),
            "q": curr_q.item(),
            "policy_loss": ploss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha,
        }

    def reset_eligibility_traces(self) -> None:
        """Reset eligibility traces."""
        for et in self.eligibility_traces_q:
            et.zero_()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.off_wandb = True
        args.render = 0

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    exp_name = f"AVG_{args.exp_name}"
    wandb.init(project="rl_practice", config=vars(args), name=exp_name, save_code=True)

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
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_{exp_name}"
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

    start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{result_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"AVG-_seed-{seed}")

    logger.info("Command line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    #### Reproducibility
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ####

    # Learner
    logger.info(f"{env.observation_space=}")
    logger.info(f"{env.action_space=}")
    args.obs_dim = env.observation_space.shape[0]

    agent = AVG(args, env, device)

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
        sim_action = sim_action * action_scale + action_bias

        # Receive reward and next state
        next_obs, reward, terminated, truncated, _ = env.step(sim_action)

        # render
        rgb_array = env.render()
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("CarRacing", bgr_array)
        cv2.waitKey(1)

        reward_normed = reward_processor.normalize(reward)
        if episode_id % args.record_interval_episode == 0:
            save_image_dir = result_dir / f"images/{episode_id:06d}"
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

        if total_step % 100 == 0:
            step_data = {
                "global_step": total_step,
                "losses/actor_loss": stats["policy_loss"],
                "losses/qf1_values": stats["q"],
                "losses/alpha": stats["alpha"],
                "losses/alpha_loss": stats["alpha_loss"],
            }
            wandb.log(step_data)

        # Termination
        if terminated or truncated:
            curr_ave_delta = sum_delta / ep_step
            curr_ave_lprob = sum_lprob / ep_step
            duration_total_sec = int(time.time() - start_time)

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
            df.to_csv(f"{result_dir}/result.tsv", index=False, sep="\t")
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
            sum_lprob = 0
            sum_reward_normed = 0
            episode_id += 1
