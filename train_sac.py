# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tyro
from torch import optim
from tqdm import tqdm

import wandb
from network import Actor, SoftQNetwork


@dataclass
class ReplayBufferData:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        size: int,
        obs_shape: np.ndarray,
        action_shape: np.ndarray,
        device: torch.device,
    ) -> None:
        self.size = size
        self.action_shape = action_shape
        self.device = device

        self.observations = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        self.observations[self.idx] = obs
        self.next_observations[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int) -> ReplayBufferData:
        idx = np.random.randint(0, self.size if self.full else self.idx, size=batch_size)
        return ReplayBufferData(
            torch.Tensor(self.observations[idx]).to(self.device),
            torch.Tensor(self.next_observations[idx]).to(self.device),
            torch.Tensor(self.actions[idx]).to(self.device),
            torch.Tensor(self.rewards[idx]).to(self.device),
            torch.Tensor(self.dones[idx]).to(self.device),
        )


@dataclass
class Args:
    exp_name: str = ""
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    gpu_id: int = 0
    """the gpu id to use"""

    # Algorithm specific arguments
    env_id: str = "CarRacing-v3"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"SAC_{args.env_id}_{args.exp_name}"

    wandb.init(
        project="cleanRL",
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.cuda.set_device(args.gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).resolve().parent / "results" / f"{datetime_str}_SAC"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_step = []
    log_episode = []

    # env setup
    env = gym.make(args.env_id, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.Autoreset(env)
    env.action_space.seed(args.seed)

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env, hidden_dim=256, use_normalize=False).to(device)
    qf1 = SoftQNetwork(env, hidden_dim=256, use_normalize=False).to(device)
    qf2 = SoftQNetwork(env, hidden_dim=256, use_normalize=False).to(device)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

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

        # execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        rb.add(obs, next_obs, action, reward, termination or truncation)

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
            log_step_df.to_csv(result_dir / "log_step.tsv", sep="\t", index=False)

    env.close()
