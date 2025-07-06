import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.diffusion_policy import DiffusionPolicy, DiffusionStatePredictor, TimestepEmbedder
from networks.sequence_processor import SequenceProcessor


def create_sequence_tokens(observations, rewards, actions, sequence_module, encoder_image):
    """Create interleaved sequence tokens from observations, rewards, and actions"""
    batch_size, seq_len = observations.shape[:2]

    # Encode all states at once
    states = observations.view(batch_size * seq_len, *observations.shape[2:])
    states = encoder_image.encode(states)
    states = states.view(batch_size, seq_len, sequence_module.cnn_dim)

    # Encode all rewards at once
    rewards = sequence_module.encoder_reward(rewards)
    rewards = rewards.view(batch_size, seq_len, sequence_module.reward_dim)

    # Encode all actions at once
    actions = sequence_module.encoder_action(actions)
    actions = actions.view(batch_size, seq_len, sequence_module.token_dim)

    # Create state+reward tokens
    state_reward_tokens = torch.cat([states, rewards], dim=-1)

    # Stack and interleave tokens
    stacked_tokens = torch.stack([state_reward_tokens, actions], dim=2)
    sequence_tensor = stacked_tokens.view(batch_size, seq_len * 2, sequence_module.token_dim)
    sequence_tensor = sequence_tensor[:, :-1]  # Remove last token

    return sequence_tensor, states


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
    sequence_tensor, _ = create_sequence_tokens(
        seq_obs_tensor,
        seq_reward_tensor,
        seq_action_tensor,
        network.sequence_model,
        network.encoder_image,
    )

    # Process and predict
    processed_sequence = network.sequence_model.sequence_processor(sequence_tensor)
    # The last token should be an action token (odd index), use it to predict next state+reward
    last_action_token = processed_sequence[:, -2]
    pred_state, _, _ = network.sequence_model.state_predictor.get_state(last_action_token)

    # Compare prediction with actual next observation
    pred_obs = network.encoder_image.decode(pred_state)
    pred_obs_np = pred_obs[0].detach().cpu().numpy().transpose(1, 2, 0)

    # Store prediction data for visualization
    # concat_images expects [0, 1] range, will multiply by 255 internally
    pred_obs_float = np.clip(pred_obs_np, 0, 1)

    # Convert next_obs to [0, 1] format to match pred_obs_float
    current_obs_float = next_obs.transpose(1, 2, 0)

    return current_obs_float, pred_obs_float


class SequenceModelingModule(nn.Module):
    def __init__(self, cnn_dim, action_dim, seq_len, args):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.action_dim = action_dim
        self.reward_dim = 32
        self.token_dim = self.cnn_dim + self.reward_dim

        self.encoder_reward = TimestepEmbedder(self.reward_dim)
        self.encoder_action = nn.Linear(action_dim, self.token_dim)
        self.sequence_processor = SequenceProcessor(
            seq_len=seq_len,
            hidden_dim=self.token_dim,
            sparsity=args.sparsity,
        )
        self.state_predictor = DiffusionStatePredictor(
            input_dim=self.token_dim,
            state_dim=self.cnn_dim,
            hidden_dim=args.predictor_hidden_dim,
            block_num=args.predictor_block_num,
            sparsity=args.sparsity,
        )
        self.reward_predictor = nn.Linear(self.token_dim, 1)
        self.action_predictor = DiffusionPolicy(
            state_dim=self.token_dim,
            action_dim=action_dim,
            hidden_dim=args.actor_hidden_dim,
            block_num=args.actor_block_num,
            sparsity=args.sparsity,
        )

    def compute_sequence_loss(self, data, encoder_image):
        sequence_tensor, encoded_states = create_sequence_tokens(
            data.observations, data.rewards, data.actions, self, encoder_image
        )

        processed_sequence = self.sequence_processor(sequence_tensor)
        seq_len_tokens = processed_sequence.shape[1]

        state_reward_positions = torch.arange(
            0, seq_len_tokens, 2, device=processed_sequence.device
        )
        action_positions = torch.arange(1, seq_len_tokens, 2, device=processed_sequence.device)

        # 各損失
        action_loss = 0.0
        state_loss = 0.0
        reward_loss = 0.0

        #####################
        # Action prediction #
        #####################
        valid_state_positions = state_reward_positions[state_reward_positions < seq_len_tokens - 1]
        state_reward_tokens = processed_sequence[:, valid_state_positions]
        batch_size, n_positions, _ = state_reward_tokens.shape

        flat_tokens = state_reward_tokens.view(batch_size * n_positions, self.token_dim)
        target_actions = data.actions[:, valid_state_positions // 2]
        target_actions_flat = target_actions.view(batch_size * n_positions, self.action_dim)

        # Flow Matching for action prediction
        x_0 = torch.randn_like(target_actions_flat)
        t = torch.rand(size=(target_actions_flat.shape[0], 1), device=target_actions_flat.device)

        # Sample from interpolation path
        x_t = (1.0 - t) * x_0 + t * target_actions_flat

        # Predict velocity using forward pass with x_t and t
        pred_actions_dict = self.action_predictor.forward(x_t, t.squeeze(1), flat_tokens)
        pred_actions = pred_actions_dict["output"]

        # Conditional vector field
        u_t = target_actions_flat - x_0

        # Flow Matching loss
        action_loss = F.mse_loss(pred_actions, u_t)

        ###########################
        # State+reward prediction #
        ###########################
        valid_action_positions = action_positions[action_positions < seq_len_tokens - 1]
        action_tokens = processed_sequence[:, valid_action_positions]
        batch_size, n_positions, _ = action_tokens.shape

        flat_tokens = action_tokens.view(batch_size * n_positions, self.token_dim)

        state_indices = (valid_action_positions + 1) // 2
        current_states = encoded_states[:, state_indices]
        current_rewards = data.rewards[:, state_indices]

        # 現在のstateとrewardを予測
        target_states_flat = current_states.view(batch_size * n_positions, self.cnn_dim)
        target_rewards_flat = current_rewards.view(batch_size * n_positions, 1)

        # State prediction using Flow Matching (拡散)
        x_0_state = torch.randn_like(target_states_flat)
        t_state = torch.rand(
            size=(target_states_flat.shape[0], 1), device=target_states_flat.device
        )

        # Sample from interpolation path for state
        x_t_state = (1.0 - t_state) * x_0_state + t_state * target_states_flat

        # Predict velocity for state using forward pass
        pred_state_dict = self.state_predictor.forward(x_t_state, t_state.squeeze(1), flat_tokens)
        pred_states_flat = pred_state_dict["output"]

        # Conditional vector field for state
        u_t_state = target_states_flat - x_0_state

        # Flow Matching loss for state
        state_loss = F.mse_loss(pred_states_flat, u_t_state)

        # Reward prediction using simple Linear regression
        pred_rewards_flat = self.reward_predictor(flat_tokens)

        # Simple MSE loss for reward prediction
        reward_loss = F.mse_loss(pred_rewards_flat, target_rewards_flat)

        # Total sequence loss
        seq_loss = action_loss + state_loss + reward_loss * 0.1

        activations_dict = {}

        info_dict = {
            "seq_loss": seq_loss.item(),
            "action_loss": action_loss.item(),
            "state_loss": state_loss.item(),
            "reward_loss": reward_loss.item(),
        }

        return seq_loss, activations_dict, info_dict


class SequenceModelingHelper:
    def __init__(self, seq_len, device):
        self.seq_len = seq_len
        self.device = device
        self.input_reward_list = None
        self.input_obs_list = None
        self.input_action_list = None

    def initialize_lists(self, action_dim):
        """Initialize sequence lists"""
        self.input_reward_list = [
            torch.zeros((1, 1), device=self.device) for _ in range(self.seq_len)
        ]
        self.input_obs_list = [
            torch.zeros((1, 3, 96, 96), device=self.device) for _ in range(self.seq_len)
        ]
        self.input_action_list = [
            torch.zeros((1, action_dim), device=self.device) for _ in range(self.seq_len)
        ]

    def update_lists(self, obs_tensor, reward, action):
        """Update sequence lists with new observations, rewards, and actions"""
        if self.input_obs_list is not None:
            self.input_obs_list.append(obs_tensor)
            self.input_obs_list.pop(0)

        if self.input_reward_list is not None:
            self.input_reward_list.append(torch.Tensor([[reward]]).to(self.device))
            self.input_reward_list.pop(0)

        if self.input_action_list is not None:
            self.input_action_list.append(torch.Tensor(action).to(self.device).unsqueeze(0))
            self.input_action_list.pop(0)

    def predict_next_state(self, action, next_obs, network):
        """Predict next state using sequence modeling"""
        if self.input_obs_list is None:
            return None, None

        return predict_next_state(
            self.input_obs_list,
            self.input_reward_list,
            self.input_action_list,
            action,
            next_obs,
            network,
            self.device,
            self.seq_len,
        )
