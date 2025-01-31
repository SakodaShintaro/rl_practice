# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch import optim
from tqdm import tqdm

import wandb
from network import Actor, SoftQNetwork


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

    # Algorithm specific arguments
    env_id: str = "Humanoid-v5"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.Autoreset(env)
    env.action_space.seed(args.seed)

    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(env, use_normalize=False).to(device)
    qf1 = SoftQNetwork(env, use_normalize=False).to(device)
    qf2 = SoftQNetwork(env, use_normalize=False).to(device)
    qf1_target = SoftQNetwork(env, use_normalize=False).to(device)
    qf2_target = SoftQNetwork(env, use_normalize=False).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        n_envs=1,
        handle_timeout_termination=False,
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
        rb.add(obs, next_obs, action, reward, termination or truncation, info)

        if termination or truncation:
            data_dict = {
                "global_step": global_step,
                "episodic_return": info["episode"]["r"],
                "episodic_length": info["episode"]["l"],
            }
            wandb.log(data_dict)
            obs, _ = env.reset()
        else:
            obs = next_obs

        # training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
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

            # update the target networks
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

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
                }
                wandb.log(data_dict)

        progress_bar.update(1)

    env.close()
