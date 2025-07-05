# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from hl_gauss_pytorch import HLGaussLoss
from torch import optim
from tqdm import tqdm

import wandb
from metrics.compute_norm import compute_gradient_norm, compute_parameter_norm
from metrics.statistical_metrics_computer import StatisticalMetricsComputer
from networks.backbone import AE
from networks.diffusion_policy import (
    DiffusionPolicy,
    DiffusionStateRewardPredictor,
    TimestepEmbedder,
)
from networks.sac_tanh_policy_and_q import SacQ
from networks.sequence_processor import SequenceProcessor
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer
from utils import concat_images
from wrappers import make_env


def create_sequence_tokens(observations, rewards, actions, network, device):
    """Create interleaved sequence tokens from observations, rewards, and actions"""
    batch_size, seq_len = observations.shape[:2]
    cnn_dim = network.cnn_dim  # 576
    reward_dim = 32
    token_dim = cnn_dim + reward_dim  # 608

    # Encode all states at once
    all_states = observations.view(batch_size * seq_len, *observations.shape[2:])
    all_state_encs = network.encoder_image.encode(all_states)
    all_state_encs = all_state_encs.view(batch_size, seq_len, cnn_dim)

    # Encode all rewards at once
    all_reward_encs = network.encoder_reward(rewards.view(batch_size * seq_len))
    all_reward_encs = all_reward_encs.view(batch_size, seq_len, reward_dim)

    # Encode all actions at once
    all_actions = actions.view(batch_size * seq_len, actions.shape[-1])
    all_action_encs = network.encoder_action(all_actions)
    all_action_encs = all_action_encs.view(batch_size, seq_len, token_dim)

    # Create state+reward tokens
    state_reward_tokens = torch.cat([all_state_encs, all_reward_encs], dim=-1)

    # Stack and interleave tokens
    stacked_tokens = torch.stack([state_reward_tokens, all_action_encs], dim=2)
    sequence_tensor = stacked_tokens.view(batch_size, seq_len * 2, token_dim)
    sequence_tensor = sequence_tensor[:, :-1]  # Remove last token

    return sequence_tensor


def predict_next_state(
    input_obs_list, input_reward_list, input_action_list, action, next_obs, network, device, seq_len
):
    """Predict next state and prepare visualization images"""
    # Prepare sequence data
    seq_obs_tensor = torch.stack(input_obs_list, dim=1)
    seq_reward_tensor = torch.stack(input_reward_list, dim=1)
    current_action_tensor = torch.Tensor(action).to(device).unsqueeze(0)
    seq_action_list = input_action_list[-seq_len + 1 :] + [current_action_tensor]
    seq_action_tensor = torch.stack(seq_action_list, dim=1)

    # Create sequence tokens using shared function
    sequence_tensor = create_sequence_tokens(
        seq_obs_tensor, seq_reward_tensor, seq_action_tensor, network, device
    )

    # Process and predict
    processed_sequence = network.sequence_processor(sequence_tensor)
    # The last token should be an action token (odd index), use it to predict next state+reward
    last_action_token = processed_sequence[:, -1]
    pred_state_reward, _, _ = network.state_reward_predictor.get_state_reward(last_action_token)

    # Compare prediction with actual next observation
    pred_state = pred_state_reward[:, : network.cnn_dim]
    pred_obs = network.encoder_image.decode(pred_state)
    pred_obs_np = pred_obs[0].detach().cpu().numpy().transpose(1, 2, 0)

    # Store prediction data for visualization
    # concat_images expects [0, 1] range, will multiply by 255 internally
    pred_obs_float = np.clip(pred_obs_np, 0, 1)

    # Convert next_obs to [0, 1] format to match pred_obs_float
    current_obs_float = next_obs.transpose(1, 2, 0)

    return current_obs_float, pred_obs_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer_size", type=int, default=int(2e4))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=4000)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--actor_block_num", type=int, default=1)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--critic_block_num", type=int, default=1)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--apply_masks_during_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_weight_projection", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


class Network(nn.Module):
    def __init__(self, num_bins: int, sparsity: float, action_dim: int, seq_len: int):
        super(Network, self).__init__()
        self.sparsity = sparsity
        cnn_dim = 4 * 12 * 12  # 576
        self.cnn_dim = cnn_dim
        reward_dim = 32
        token_dim = cnn_dim + reward_dim  # 608
        self.encoder_image = AE()
        self.encoder_reward = TimestepEmbedder(reward_dim)
        self.encoder_action = nn.Linear(action_dim, token_dim)
        self.sequence_processor = SequenceProcessor(
            seq_len=seq_len,
            hidden_dim=token_dim,
            sparsity=args.sparsity,
        )
        self.actor = DiffusionPolicy(
            state_dim=cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.qf1 = SacQ(
            in_channels=cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=num_bins,
            sparsity=args.sparsity,
        )
        self.qf2 = SacQ(
            in_channels=cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=num_bins,
            sparsity=args.sparsity,
        )
        self.state_reward_predictor = DiffusionStateRewardPredictor(
            input_dim=token_dim,
            state_dim=cnn_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.action_predictor = DiffusionPolicy(
            state_dim=token_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.off_wandb = True
        args.learning_starts = 10
        args.render = 0

    if args.off_wandb:
        os.environ["WANDB_MODE"] = "offline"

    exp_name = f"SAC_{args.exp_name}"
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

    action_dim = np.prod(env.action_space.shape)
    seq_len = 2
    assert seq_len >= 2, "seq_len must be >= 2 for sequence modeling"
    num_bins = 51
    network = Network(
        num_bins=num_bins, sparsity=args.sparsity, action_dim=action_dim, seq_len=seq_len
    ).to(device)
    lr = 1e-4
    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-5)
    hl_gauss_loss = HLGaussLoss(
        min_value=-30,
        max_value=+30,
        num_bins=num_bins,
        clamp_to_range=True,
    ).to(device)

    rb = ReplayBuffer(
        args.buffer_size,
        seq_len,
        env.observation_space.shape,
        env.action_space.shape,
        device,
    )

    # Initialize gradient norm targets
    monitoring_targets = {
        "total": network,
        "sequence_processor": network.sequence_processor,
        "actor": network.actor,
        "qf1": network.qf1,
        "qf2": network.qf2,
        "action_predictor": network.action_predictor,
        "state_reward_predictor": network.state_reward_predictor,
    }

    # Initialize weight projection if enabled
    weight_projection_norms = {}
    if args.use_weight_projection:
        weight_projection_norms["sequence_processor"] = get_initial_norms(
            network.sequence_processor
        )
        weight_projection_norms["actor"] = get_initial_norms(network.actor)
        weight_projection_norms["qf1"] = get_initial_norms(network.qf1)
        weight_projection_norms["qf2"] = get_initial_norms(network.qf2)
        weight_projection_norms["action_predictor"] = get_initial_norms(network.action_predictor)
        weight_projection_norms["state_reward_predictor"] = get_initial_norms(
            network.state_reward_predictor
        )

    start_time = time.time()

    # start the game
    global_step = 0
    score_list = []
    obs, _ = env.reset(seed=seed)
    progress_bar = tqdm(range(args.learning_starts), dynamic_ncols=True)
    curr_image_dir = None
    step_limit = 200_000

    # Initialize dummy prediction images
    curr_obs_float = np.zeros((96, 96, 3), dtype=np.float32)
    pred_obs_float = np.zeros((96, 96, 3), dtype=np.float32)
    metrics_computers = {
        "state": StatisticalMetricsComputer(),
        "qf1": StatisticalMetricsComputer(),
        "qf2": StatisticalMetricsComputer(),
        "actor": StatisticalMetricsComputer(),
    }

    for episode_id in range(10000):
        if (episode_id + 1) % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{episode_id:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = env.reset()
        reward_list = []
        first_value = None

        input_reward_list = [torch.zeros((1, 1), device=device) for _ in range(seq_len)]
        input_obs_list = [torch.zeros((1, 3, 96, 96), device=device) for _ in range(seq_len)]
        input_action_list = [torch.zeros((1, action_dim), device=device) for _ in range(seq_len)]

        while True:
            global_step += 1

            # select action
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
            if global_step < args.learning_starts:
                action = env.action_space.sample()
                progress_bar.update(1)
            else:
                with torch.inference_mode():
                    input_obs_list.append(obs_tensor)
                    input_obs_list.pop(0)
                    input_reward_tensor = torch.stack(input_reward_list, dim=1)
                    input_obs_tensor = torch.stack(input_obs_list, dim=1)
                    input_action_tensor = torch.stack(input_action_list, dim=1)
                    output_enc = network.encoder_image.encode(obs_tensor)
                    action, selected_log_pi, _ = network.actor.get_action(output_enc)
                    action = action[0].detach().cpu().numpy()
                    action = action * action_scale + action_bias

                    action_noise = env.action_space.sample()
                    c = args.action_noise
                    action = (1 - c) * action + c * action_noise
                    action = np.clip(action, action_low, action_high)

                    input_action_list.append(torch.Tensor(action).to(device).unsqueeze(0))
                    input_action_list.pop(0)

            # execute the game and log data.
            next_obs, reward, termination, truncation, info = env.step(action)
            reward /= 10.0
            rb.add(obs, action, reward, termination or truncation)

            input_reward_list.append(torch.Tensor([[reward]]).to(device))
            input_reward_list.pop(0)

            # Sequence processing for next state prediction (after getting next_obs)
            if global_step > args.learning_starts:
                curr_obs_float, pred_obs_float = predict_next_state(
                    input_obs_list,
                    input_reward_list,
                    input_action_list,
                    action,
                    next_obs,
                    network,
                    device,
                    seq_len,
                )

            # render
            if args.render:
                bgr_array = concat_images(env.render(), curr_obs_float, pred_obs_float)
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)

            # save images for specific episodes
            if episode_id % image_save_interval == 0 and curr_image_dir is not None:
                bgr_array = concat_images(env.render(), curr_obs_float, pred_obs_float)
                cv2.imwrite(str(curr_image_dir / f"{global_step:08d}.png"), bgr_array)

            if termination or truncation:
                break

            if global_step >= step_limit:
                break

            obs = next_obs

            if global_step <= args.learning_starts:
                continue

            # training.
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                state_next = network.encoder_image.encode(data.observations[:, -1])
                next_state_actions, next_state_log_pi, _ = network.actor.get_action(state_next)
                qf1_next_output_dict = network.qf1(state_next, next_state_actions)
                qf2_next_output_dict = network.qf2(state_next, next_state_actions)
                qf1_next_target = qf1_next_output_dict["output"]
                qf2_next_target = qf2_next_output_dict["output"]
                qf1_next_target = hl_gauss_loss(qf1_next_target).unsqueeze(-1)
                qf2_next_target = hl_gauss_loss(qf2_next_target).unsqueeze(-1)
                min_q = torch.min(qf1_next_target, qf2_next_target)
                min_qf_next_target = min_q.view(-1)
                curr_reward = data.rewards[:, -2].flatten()
                curr_continue = 1 - data.dones[:, -2].flatten()
                next_q_value = curr_reward + curr_continue * args.gamma * min_qf_next_target

            state_curr = network.encoder_image.encode(data.observations[:, -2])

            # Get Q-values and activations for Srank computation
            qf1_output_dict = network.qf1(state_curr, data.actions[:, -2])
            qf2_output_dict = network.qf2(state_curr, data.actions[:, -2])

            qf1_a_values = qf1_output_dict["output"]
            qf2_a_values = qf2_output_dict["output"]

            qf1_loss = hl_gauss_loss(qf1_a_values, next_q_value)
            qf2_loss = hl_gauss_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            pi, log_pi, _ = network.actor.get_action(state_curr)
            for param in network.qf1.parameters():
                param.requires_grad_(False)
            for param in network.qf2.parameters():
                param.requires_grad_(False)
            qf1_pi_output_dict = network.qf1(state_curr, pi)
            qf2_pi_output_dict = network.qf2(state_curr, pi)
            qf1_pi = qf1_pi_output_dict["output"]
            qf2_pi = qf2_pi_output_dict["output"]
            qf1_pi = hl_gauss_loss(qf1_pi).unsqueeze(-1)
            qf2_pi = hl_gauss_loss(qf2_pi).unsqueeze(-1)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = -min_qf_pi.mean()
            for param in network.qf1.parameters():
                param.requires_grad_(True)
            for param in network.qf2.parameters():
                param.requires_grad_(True)

            # DACER2 (https://arxiv.org/abs/2505.23426) loss
            actions = pi.clone().detach()
            actions.requires_grad = True
            eps = 1e-4
            t = (torch.rand((args.batch_size, 1), device=device)) * (1 - eps) + eps
            c = 0.4
            d = -1.8
            w_t = torch.exp(c * t + d)

            def calc_target(q_network, actions):
                q_output_dict = q_network(state_curr, actions)
                q_values = q_output_dict["output"]
                q_values = hl_gauss_loss(q_values).unsqueeze(-1)
                q_grad = torch.autograd.grad(
                    outputs=q_values.sum(),
                    inputs=actions,
                    create_graph=True,
                )[0]
                with torch.no_grad():
                    # target = -actions / (1 - t) - t / (1 - t) * q_grad
                    target = (1 - t) / t * q_grad + 1 / t * actions
                    target /= target.norm(dim=1, keepdim=True) + 1e-8
                    return w_t * target

            target1 = calc_target(network.qf1, actions)
            target2 = calc_target(network.qf2, actions)
            target = (target1 + target2) / 2.0
            noise = torch.randn_like(actions)
            noise = torch.clamp(noise, -3.0, 3.0)
            a_t = (1.0 - t) * noise + t * actions
            actor_output_dict = network.actor.forward(a_t, t.squeeze(1), state_curr)
            v = actor_output_dict["output"]
            dacer_loss = F.mse_loss(v, target)
            actor_loss += dacer_loss * 0.05

            # Sequence modeling loss
            # Get dimensions
            cnn_dim = network.cnn_dim
            reward_dim = 32
            token_dim = cnn_dim + reward_dim

            # Create sequence tokens using shared function
            sequence_tensor = create_sequence_tokens(
                data.observations, data.rewards, data.actions, network, device
            )

            # Process through sequence processor
            processed_sequence = network.sequence_processor(sequence_tensor)

            # Vectorized prediction
            seq_len_tokens = processed_sequence.shape[1]

            # Action prediction: predict actions from state+reward tokens
            state_reward_positions = torch.arange(0, seq_len_tokens, 2, device=device)
            action_positions = torch.arange(1, seq_len_tokens, 2, device=device)
            seq_loss = 0.0

            # Predict actions from state+reward tokens
            if len(state_reward_positions) > 0 and len(action_positions) > 0:
                # Get state+reward tokens that have corresponding actions
                valid_state_positions = state_reward_positions[
                    state_reward_positions < seq_len_tokens - 1
                ]
                if len(valid_state_positions) > 0:
                    # Batch process action predictions
                    state_reward_tokens = processed_sequence[
                        :, valid_state_positions
                    ]  # (batch, n_positions, token_dim)
                    batch_size, n_positions, _ = state_reward_tokens.shape

                    # Flatten for batch processing
                    flat_tokens = state_reward_tokens.view(batch_size * n_positions, token_dim)
                    pred_actions, _, _ = network.action_predictor.get_action(flat_tokens)

                    # Reshape back and get target actions
                    pred_actions = pred_actions.view(batch_size, n_positions, action_dim)
                    target_actions = data.actions[:, valid_state_positions // 2]

                    seq_loss += F.mse_loss(pred_actions, target_actions)

            # Predict state+reward from action tokens
            if len(action_positions) > 0:
                # Get action tokens that have corresponding next states
                valid_action_positions = action_positions[action_positions < seq_len_tokens - 1]
                if len(valid_action_positions) > 0:
                    # Batch process state+reward predictions
                    action_tokens = processed_sequence[
                        :, valid_action_positions
                    ]  # (batch, n_positions, token_dim)
                    batch_size, n_positions, _ = action_tokens.shape

                    # Flatten for batch processing
                    flat_tokens = action_tokens.view(batch_size * n_positions, token_dim)
                    pred_state_rewards, _, _ = network.state_reward_predictor.get_state_reward(
                        flat_tokens
                    )

                    # Reshape back and get target state+rewards
                    pred_state_rewards = pred_state_rewards.view(
                        batch_size, n_positions, cnn_dim + 1
                    )

                    # Build target state+rewards
                    state_indices = (valid_action_positions + 1) // 2
                    target_states = network.encoder_image.encode(
                        data.observations[:, state_indices].view(
                            batch_size * n_positions, *data.observations.shape[2:]
                        )
                    )
                    target_states = target_states.view(batch_size, n_positions, cnn_dim)
                    target_rewards = data.rewards[:, state_indices]
                    target_state_rewards = torch.cat([target_states, target_rewards], dim=-1)

                    seq_loss += F.mse_loss(pred_state_rewards, target_state_rewards)

            # Compute srank for key activations (using intermediate layer outputs)
            feature_dict = {
                "state": state_curr,
                "actor": actor_output_dict["activation"],
                "qf1": qf1_pi_output_dict["activation"],
                "qf2": qf2_pi_output_dict["activation"],
            }

            # optimize the model
            loss = actor_loss + qf_loss + seq_loss
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)

            # Compute gradient and parameter norms
            grad_metrics = {
                key: compute_gradient_norm(model) for key, model in monitoring_targets.items()
            }
            param_metrics = {
                key: compute_parameter_norm(model) for key, model in monitoring_targets.items()
            }
            activation_norms = {
                key: value.norm(dim=1).mean().item() for key, value in feature_dict.items()
            }

            optimizer.step()

            # Apply weight projection after optimizer step
            if args.use_weight_projection:
                weight_project(
                    network.sequence_processor, weight_projection_norms["sequence_processor"]
                )
                weight_project(network.actor, weight_projection_norms["actor"])
                weight_project(network.qf1, weight_projection_norms["qf1"])
                weight_project(network.qf2, weight_projection_norms["qf2"])
                weight_project(
                    network.action_predictor, weight_projection_norms["action_predictor"]
                )
                weight_project(
                    network.state_reward_predictor,
                    weight_projection_norms["state_reward_predictor"],
                )

            # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
            if args.apply_masks_during_training:
                apply_masks_during_training(network.sequence_processor)
                apply_masks_during_training(network.actor)
                apply_masks_during_training(network.qf1)
                apply_masks_during_training(network.qf2)
                apply_masks_during_training(network.action_predictor)
                apply_masks_during_training(network.state_reward_predictor)

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
                    "losses/log_pi": log_pi.mean().item(),
                    "a_logp": selected_log_pi.mean().item(),
                    "charts/elapse_time_sec": elapsed_time,
                    "charts/SPS": global_step / elapsed_time,
                    "reward": reward,
                }

                # Add gradient norm metrics
                for key, value in grad_metrics.items():
                    data_dict[f"gradients/{key}"] = value

                # Add parameter norm metrics
                for key, value in param_metrics.items():
                    data_dict[f"parameters/{key}"] = value

                # Add activation norms
                for key, value in activation_norms.items():
                    data_dict[f"activation_norms/{key}"] = value

                # Trigger statistical metrics computation
                for feature_name, feature in feature_dict.items():
                    result_dict = metrics_computers[feature_name](feature)
                    for key, value in result_dict.items():
                        data_dict[f"{key}/{feature_name}"] = value

                data_dict["losses/dacer_loss"] = dacer_loss.item()
                data_dict["losses/seq_loss"] = (
                    seq_loss.item() if isinstance(seq_loss, torch.Tensor) else seq_loss
                )
                wandb.log(data_dict)

                fixed_data = {
                    k.replace("losses/", "").replace("charts/", ""): v for k, v in data_dict.items()
                }
                log_step.append(fixed_data)
                log_step_df = pd.DataFrame(log_step)
                log_step_df.to_csv(
                    result_dir / "log_step.tsv", sep="\t", index=False, float_format="%.3f"
                )

        if global_step >= step_limit:
            break

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
