import argparse
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gymnasium.wrappers import NormalizeObservation
from torch import nn
from torch.distributions import MultivariateNormal


def orthogonal_weight_init(m: nn.Module) -> None:
    """Orthogonal weight initialization for neural networks."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    """Continous MLP Actor for Soft Actor-Critic."""

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, n_hid: int) -> None:
        super().__init__()
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        # Two hidden layers
        self.phi = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Actor network."""
        phi = self.phi(obs.to(self.device))
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))
        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)

        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!!
        #   A mistake here can break learning :/
        action = torch.tanh(action_pre)
        action_info = {
            "mu": mu,
            "log_std": log_std,
            "dist": dist,
            "lprob": lprob,
            "action_pre": action_pre,
        }

        return action, action_info


class Q(nn.Module):
    """Continuous Q-network for Soft Actor-Critic."""

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, n_hid: int) -> None:
        super().__init__()
        self.device = device

        # Two hidden layers
        self.phi = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )
        self.q = nn.Linear(n_hid, 1)
        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Q network."""
        x = torch.cat((obs, action), -1).to(self.device)
        phi = self.phi(x)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        return self.q(phi).view(-1)


class AVG:
    """AVG Agent."""

    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.steps = 0

        self.actor = Actor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_actor,
        )
        self.Q = Q(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_critic,
        )

        self.popt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.betas)
        self.qopt = torch.optim.Adam(self.Q.parameters(), lr=cfg.critic_lr, betas=cfg.betas)

        self.alpha, self.gamma, self.device = cfg.alpha_lr, cfg.gamma, cfg.device

    def compute_action(self, obs: np.ndarray) -> tuple[torch.Tensor, dict]:
        """Compute the action and action information given an observation."""
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, action_info = self.actor(obs)
        return action, action_info

    def update(
        self,
        obs: np.ndarray,
        action: torch.Tensor,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        **kwargs: dict,
    ) -> None:
        """Update the actor and critic networks based on the observed transition."""
        obs = torch.Tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs = torch.Tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, lprob = action.to(self.device), kwargs["lprob"]

        #### Q loss
        q = self.Q(obs, action.detach())  # N.B: Gradient should NOT pass through action here
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs)
            next_lprob = action_info["lprob"]
            q2 = self.Q(next_obs, next_action)
            target_V = q2 - self.alpha * next_lprob

        delta = reward + (1 - done) * self.gamma * target_V - q
        qloss = delta**2
        ####

        # Policy loss
        ploss = self.alpha * lprob - self.Q(obs, action)  # N.B: USE reparametrized action
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        self.qopt.step()

        self.steps += 1

    def save(self, model_dir: str, unique_str: str) -> None:
        """Save the model parameters to a file."""
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
        }
        torch.save(model, f"{model_dir}/{unique_str}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Humanoid-v4", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generator")
    parser.add_argument("--N", default=2000000, type=int, help="# total timesteps for the run")
    # SAVG params
    parser.add_argument("--actor_lr", default=0.0063, type=float, help="Actor step size")
    parser.add_argument("--critic_lr", default=0.0087, type=float, help="Critic step size")
    parser.add_argument("--beta1", default=0.0, type=float, help="Beta1 parameter of Adam")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--alpha_lr", default=0.07, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument("--l2_actor", default=0, type=float, help="L2 Regularization")
    parser.add_argument("--l2_critic", default=0, type=float, help="L2 Regularization")
    parser.add_argument("--nhid_actor", default=256, type=int)
    parser.add_argument("--nhid_critic", default=256, type=int)
    # Miscellaneous
    parser.add_argument("--checkpoint", default=500000, type=int, help="Checkpoint interval")
    parser.add_argument("--save_dir", default="./results", type=Path, help="Location to store")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--n_eval", default=0, type=int, help="Number of eval episodes")
    args = parser.parse_args()

    # Adam
    args.betas = [args.beta1, 0.999]

    # CPU/GPU use for the run
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = args.save_dir / datetime_str
    save_dir.mkdir(exist_ok=True, parents=True)

    # Start experiment
    # N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow.
    # Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    tic = time.time()
    with (save_dir / "info.txt").open("w") as f:
        f.write(f"AVG-{args.env}_seed-{args.seed}\n")

    # Env
    env = gym.make(args.env, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)
    env = NormalizeObservation(env)

    #### Reproducibility
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Learner
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    agent = AVG(args)

    action_coeff = (env.action_space.high - env.action_space.low) / 2

    # Interaction
    rets, ep_steps = [], []
    ret, ep_step = 0, 0
    terminated, truncated = False, False
    obs, _ = env.reset()
    data_list = []
    try:
        for total_step in range(1, args.N + 1):
            ep_id = len(rets)

            # N.B: Action is a torch.Tensor
            action, action_info = agent.compute_action(obs)
            sim_action = action.detach().cpu().view(-1).numpy()
            sim_action *= action_coeff

            # Receive reward and next state
            next_obs, reward, terminated, truncated, _ = env.step(sim_action)
            if ep_id % 100 == 0:
                save_image_dir = save_dir / f"images/{ep_id:06d}"
                save_image_dir.mkdir(exist_ok=True, parents=True)
                image = env.render()
                cv2.imwrite(str(save_image_dir / f"{ep_step:08d}.png"), image)
            agent.update(obs, action, next_obs, reward, terminated, **action_info)
            ret += reward
            ep_step += 1

            obs = next_obs

            if total_step % args.checkpoint == 0:
                agent.save(
                    model_dir=save_dir,
                    unique_str=f"model_{total_step:010d}",
                )

            # Termination
            if terminated or truncated:
                rets.append(ret)
                ep_steps.append(ep_step)
                duration = time.time() - tic
                print(
                    f"Episode: {ep_id}| "
                    f"D: {duration:.3f} sec| "
                    f"S: {ep_step}| "
                    f"R: {ret:.2f}| "
                    f"T: {total_step}",
                )

                data_list.append(
                    {"episode_id": ep_id, "duration": duration, "steps": ep_step, "return": ret},
                )

                df = pd.DataFrame(data_list)
                df.to_csv(f"{save_dir}/result.tsv", index=False, sep="\t")

                obs, _ = env.reset()
                ret, ep_step = 0, 0
    except Exception as e:
        print(e)
        print("Exiting this run, storing partial logs in the database for future debugging...")
        traceback.print_exc()

    if not (terminated or truncated):
        # N.B: We're adding a partial episode just to make plotting easier.
        # But this data point shouldn't be used.
        rets.append(ret)
        ep_steps.append(ep_step)

    # Save returns and args before exiting run
    agent.save(model_dir=save_dir, unique_str="last_model")

    df = pd.DataFrame(data_list)
    df.to_csv(f"{save_dir}/result.tsv", index=False, sep="\t")
    plt.plot(df["episode_id"], df["return"])
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(f"{save_dir}/returns.png", bbox_inches="tight", pad_inches=0.05)
