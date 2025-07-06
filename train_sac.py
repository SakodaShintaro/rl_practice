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
from networks.diffusion_policy import DiffusionPolicy
from networks.sac_tanh_policy_and_q import SacQ
from networks.sparse_utils import apply_masks_during_training
from networks.weight_project import get_initial_norms, weight_project
from replay_buffer import ReplayBuffer
from sequence_modeling import SequenceModelingHelper, SequenceModelingModule
from utils import concat_images
from wrappers import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--buffer_size", type=int, default=int(2e4))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=4000)
    parser.add_argument("--render", type=int, default=1, choices=[0, 1])
    parser.add_argument("--off_wandb", action="store_true")
    parser.add_argument("--action_noise", type=float, default=0.0)
    parser.add_argument("--actor_hidden_dim", type=int, default=512)
    parser.add_argument("--actor_block_num", type=int, default=1)
    parser.add_argument("--critic_hidden_dim", type=int, default=1024)
    parser.add_argument("--critic_block_num", type=int, default=1)
    parser.add_argument("--predictor_hidden_dim", type=int, default=1024)
    parser.add_argument("--predictor_block_num", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--apply_masks_during_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_weight_projection", action="store_true")
    parser.add_argument("--enable_sequence_modeling", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


class Network(nn.Module):
    def __init__(
        self,
        sparsity: float,
        action_dim: int,
        seq_len: int,
        args,
        enable_sequence_modeling: bool,
    ):
        super(Network, self).__init__()
        num_bins = 51
        self.gamma = 0.99
        self.sparsity = sparsity

        self.action_dim = action_dim
        self.cnn_dim = 4 * 12 * 12  # 576
        self.reward_dim = 32
        self.token_dim = self.cnn_dim + self.reward_dim  # 608

        self.encoder_image = AE()
        self.actor = DiffusionPolicy(
            state_dim=self.cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )
        self.qf1 = SacQ(
            in_channels=self.cnn_dim,
            action_dim=action_dim,
            hidden_dim=args.critic_hidden_dim,
            block_num=args.critic_block_num,
            num_bins=num_bins,
            sparsity=args.sparsity,
        )

        # Sequence modeling components (optional)
        if enable_sequence_modeling:
            self.sequence_model = SequenceModelingModule(self.cnn_dim, action_dim, seq_len, args)
        else:
            self.sequence_model = None

        self.hl_gauss_loss = HLGaussLoss(
            min_value=-30,
            max_value=+30,
            num_bins=num_bins,
            clamp_to_range=True,
        )

    def compute_critic_loss(self, data, state_curr):
        with torch.no_grad():
            state_next = self.encoder_image.encode(data.observations[:, -1])
            next_state_actions, _, _ = self.actor.get_action(state_next)
            qf1_next_output_dict = self.qf1(state_next, next_state_actions)
            qf1_next_target = qf1_next_output_dict["output"]
            qf1_next_target = self.hl_gauss_loss(qf1_next_target).unsqueeze(-1)
            min_q = qf1_next_target
            min_qf_next_target = min_q.view(-1)
            curr_reward = data.rewards[:, -2].flatten()
            curr_continue = 1 - data.dones[:, -2].flatten()
            next_q_value = curr_reward + curr_continue * self.gamma * min_qf_next_target

        qf1_output_dict = self.qf1(state_curr, data.actions[:, -2])

        qf1_a_values = qf1_output_dict["output"]

        qf1_loss = self.hl_gauss_loss(qf1_a_values, next_q_value)
        qf_loss = qf1_loss

        activations_dict = {}

        info_dict = {
            "qf1_loss": qf1_loss.item(),
            "qf1_values": qf1_a_values.mean().item(),
            "min_qf_next_target": min_qf_next_target.mean().item(),
            "next_q_value": next_q_value.mean().item(),
        }

        return qf_loss, activations_dict, info_dict

    def compute_actor_loss(self, state_curr):
        pi, log_pi, _ = self.actor.get_action(state_curr)

        for param in self.qf1.parameters():
            param.requires_grad_(False)

        qf1_pi_output_dict = self.qf1(state_curr, pi)
        qf1_pi = qf1_pi_output_dict["output"]
        qf1_pi = self.hl_gauss_loss(qf1_pi).unsqueeze(-1)
        min_qf_pi = qf1_pi
        actor_loss = -min_qf_pi.mean()

        for param in self.qf1.parameters():
            param.requires_grad_(True)

        # DACER2 loss (https://arxiv.org/abs/2505.23426)
        actions = pi.clone().detach()
        actions.requires_grad = True
        eps = 1e-4
        device = pi.device
        batch_size = pi.shape[0]
        t = (torch.rand((batch_size, 1), device=device)) * (1 - eps) + eps
        c = 0.4
        d = -1.8
        w_t = torch.exp(c * t + d)

        def calc_target(q_network, actions):
            q_output_dict = q_network(state_curr, actions)
            q_values = q_output_dict["output"]
            q_values = self.hl_gauss_loss(q_values).unsqueeze(-1)
            q_grad = torch.autograd.grad(
                outputs=q_values.sum(),
                inputs=actions,
                create_graph=True,
            )[0]
            with torch.no_grad():
                target = (1 - t) / t * q_grad + 1 / t * actions
                target /= target.norm(dim=1, keepdim=True) + 1e-8
                return w_t * target

        target1 = calc_target(self.qf1, actions)
        target = target1
        noise = torch.randn_like(actions)
        noise = torch.clamp(noise, -3.0, 3.0)
        a_t = (1.0 - t) * noise + t * actions
        actor_output_dict = self.actor.forward(a_t, t.squeeze(1), state_curr)
        v = actor_output_dict["output"]
        dacer_loss = F.mse_loss(v, target)

        # Combine actor losses
        total_actor_loss = actor_loss + dacer_loss * 0.05

        activations_dict = {
            "actor": actor_output_dict["activation"],
            "qf1": qf1_pi_output_dict["activation"],
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
            "log_pi": log_pi.mean().item(),
        }

        return total_actor_loss, activations_dict, info_dict

    def compute_sequence_loss(self, data):
        if self.sequence_model is None:
            # Return zero loss when sequence modeling is disabled
            return (
                torch.tensor(0.0, device=data.observations.device),
                {},
                {},
            )

        return self.sequence_model.compute_sequence_loss(data, self.encoder_image)


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
    if args.enable_sequence_modeling:
        assert seq_len >= 2, "seq_len must be >= 2 for sequence modeling"
    network = Network(
        sparsity=args.sparsity,
        action_dim=action_dim,
        seq_len=seq_len,
        args=args,
        enable_sequence_modeling=args.enable_sequence_modeling,
    ).to(device)
    lr = 1e-4
    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-5)

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
        "actor": network.actor,
        "qf1": network.qf1,
    }

    # Initialize weight projection if enabled
    weight_projection_norms = {}
    if args.use_weight_projection:
        weight_projection_norms["actor"] = get_initial_norms(network.actor)
        weight_projection_norms["qf1"] = get_initial_norms(network.qf1)

    if args.enable_sequence_modeling:
        monitoring_targets["sequence_model"] = network.sequence_model
        weight_projection_norms["sequence_model"] = get_initial_norms(network.sequence_model)

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

    # Initialize sequence modeling helper
    if args.enable_sequence_modeling:
        sequence_helper = SequenceModelingHelper(seq_len, device)

    metrics_computers = {
        "state": StatisticalMetricsComputer(),
        "qf1": StatisticalMetricsComputer(),
        "actor": StatisticalMetricsComputer(),
    }

    for episode_id in range(10000):
        if episode_id == 0 or (episode_id + 1) % image_save_interval == 0:
            curr_image_dir = image_dir / f"ep_{episode_id + 1:08d}"
            curr_image_dir.mkdir(parents=True, exist_ok=True)
        else:
            curr_image_dir = None

        obs, _ = env.reset()
        reward_list = []
        first_value = None

        # Initialize sequence modeling lists if enabled
        if args.enable_sequence_modeling:
            sequence_helper.initialize_lists(action_dim)

        while True:
            global_step += 1

            # select action
            obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
            if global_step < args.learning_starts:
                action = env.action_space.sample()
                progress_bar.update(1)
            else:
                with torch.inference_mode():
                    output_enc = network.encoder_image.encode(obs_tensor)
                    action, selected_log_pi, _ = network.actor.get_action(output_enc)
                    action = action[0].detach().cpu().numpy()
                    action = action * action_scale + action_bias

                    action_noise = env.action_space.sample()
                    c = args.action_noise
                    action = (1 - c) * action + c * action_noise
                    action = np.clip(action, action_low, action_high)

            # execute the game and log data.
            next_obs, reward, termination, truncation, info = env.step(action)
            reward /= 10.0
            rb.add(obs, action, reward, termination or truncation)

            # Update sequence modeling lists
            if args.enable_sequence_modeling:
                sequence_helper.update_lists(obs_tensor, reward, action)

            # Sequence processing for next state prediction (after getting next_obs)
            if (
                args.enable_sequence_modeling
                and global_step > args.learning_starts
                and global_step % 10 == 0
            ):
                curr_obs_float, pred_obs_float = sequence_helper.predict_next_state(
                    action, next_obs, network
                )

            # render
            if args.render:
                bgr_array = concat_images(env.render(), curr_obs_float, pred_obs_float)
                cv2.imshow("CarRacing", bgr_array)
                cv2.waitKey(1)

            # save images for specific episodes
            if curr_image_dir is not None:
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

            # Compute all losses using refactored methods
            state_curr = network.encoder_image.encode(data.observations[:, -2])
            qf_loss, qf_activations, qf_info = network.compute_critic_loss(data, state_curr)
            actor_loss, actor_activations, actor_info = network.compute_actor_loss(state_curr)
            seq_loss, seq_activations, seq_info = network.compute_sequence_loss(data)

            # Combine all activations for feature_dict
            feature_dict = {
                "state": state_curr,
                **actor_activations,
                **qf_activations,
                **seq_activations,
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
                weight_project(network.actor, weight_projection_norms["actor"])
                weight_project(network.qf1, weight_projection_norms["qf1"])

                if args.enable_sequence_modeling:
                    weight_project(
                        network.sequence_model, weight_projection_norms["sequence_model"]
                    )

            # Apply sparsity masks after optimizer step to ensure pruned weights stay zero
            if args.apply_masks_during_training:
                apply_masks_during_training(network.actor)
                apply_masks_during_training(network.qf1)

                if args.enable_sequence_modeling:
                    apply_masks_during_training(network.sequence_model)

            if global_step % 10 == 0:
                elapsed_time = time.time() - start_time
                data_dict = {
                    "global_step": global_step,
                    "a_logp": selected_log_pi.mean().item(),
                    "charts/elapse_time_sec": elapsed_time,
                    "charts/SPS": global_step / elapsed_time,
                    "reward": reward,
                }

                # Add loss information from compute methods
                for key, value in qf_info.items():
                    data_dict[f"losses/{key}"] = value

                for key, value in actor_info.items():
                    data_dict[f"losses/{key}"] = value

                for key, value in seq_info.items():
                    data_dict[f"losses/{key}"] = value

                # Add combined loss
                data_dict["losses/qf_loss"] = qf_loss.item() / 2.0

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

        episode_id += 1

        if is_solved:
            print(
                f"Solved! Running reward is now {recent_average_score} and the last episode runs to {score}!"
            )
            break

    env.close()
