# Derived from https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
import numpy as np
import torch
from diffusers.models import AutoencoderTiny
from gymnasium import spaces
from torch import nn, optim
from torch.distributions import Categorical
from torch.nn import functional as F


class Buffer:
    """The buffer stores and prepares the training data. It supports recurrent policies."""

    def __init__(
        self,
        worker_steps: int,
        hidden_size: int,
        layer_type: str,
        sequence_length: int,
        observation_space: spaces.Box,
        action_space_shape: tuple,
        device: torch.device,
    ) -> None:
        # Setup members
        self.device = device
        self.worker_steps = worker_steps
        self.n_mini_batches = 8
        self.batch_size = self.worker_steps
        self.layer_type = layer_type
        self.sequence_length = sequence_length
        self.true_sequence_length = 0

        # Initialize the buffer's data storage for a single environment
        self.rewards = np.zeros(self.worker_steps, dtype=np.float32)
        self.actions = torch.zeros((self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros(self.worker_steps, dtype=bool)
        self.obs = torch.zeros((self.worker_steps,) + observation_space.shape)
        self.hxs = torch.zeros((self.worker_steps, hidden_size))
        self.cxs = torch.zeros((self.worker_steps, hidden_size))
        self.log_probs = torch.zeros((self.worker_steps, len(action_space_shape)))
        self.values = torch.zeros(self.worker_steps)
        self.advantages = torch.zeros(self.worker_steps)

    def prepare_batch_dict(self) -> None:
        """Flattens the training samples and stores them inside a dictionary. Due to using a recurrent policy,
        the data is split into episodes or sequences beforehand.
        """
        samples = {
            "obs": self.obs,
            "actions": self.actions,
            "loss_mask": torch.ones(self.worker_steps, dtype=torch.bool),
        }

        samples["hxs"] = self.hxs
        if self.layer_type == "lstm":
            samples["cxs"] = self.cxs

        # Determine indices at which episodes terminate
        episode_done_indices = list(np.where(self.dones)[0])
        if not episode_done_indices or episode_done_indices[-1] != self.worker_steps - 1:
            episode_done_indices.append(self.worker_steps - 1)

        index_sequences, max_sequence_length = self._arange_sequences(
            torch.arange(self.worker_steps), episode_done_indices
        )
        self.flat_sequence_indices = np.asarray(
            [seq.tolist() for seq in index_sequences], dtype=object
        )

        for key, value in samples.items():
            value_tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            sequences, _ = self._arange_sequences(value_tensor, episode_done_indices)
            sequences = [
                self._pad_sequence(sequence, max_sequence_length) for sequence in sequences
            ]
            stacked = torch.stack(sequences, dim=0)
            if key in ("hxs", "cxs"):
                stacked = stacked[:, 0]
            samples[key] = stacked

        self.num_sequences = len(self.flat_sequence_indices)
        self.actual_sequence_length = max_sequence_length
        self.true_sequence_length = max_sequence_length

        samples["values"] = self.values
        samples["log_probs"] = self.log_probs
        samples["advantages"] = self.advantages

        self.samples_flat = {}
        for key, value in samples.items():
            if key in ("hxs", "cxs"):
                self.samples_flat[key] = value
            elif key in ("values", "log_probs", "advantages"):
                self.samples_flat[key] = value
            else:
                if value.dim() == 1:
                    self.samples_flat[key] = value
                else:
                    self.samples_flat[key] = value.reshape(
                        value.shape[0] * value.shape[1], *value.shape[2:]
                    )

    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pads a sequence to the target length using zeros.

        Arguments:
            sequence {np.ndarray} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(
                ((delta_length,) + sequence.shape[1:]),
                dtype=sequence.dtype,
                device=sequence.device,
            )
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype, device=sequence.device)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)

    def _arange_sequences(self, data, episode_done_indices):
        """Splits the provided data into episodes and then into sequences.
        The split points are indicated by the environments' done signals.

        Arguments:
            data {torch.tensor} -- The to be split data arrange into num_worker, worker_steps
            episode_done_indices {list} -- Nested list indicating the indices of done signals. Trajectory ends are treated as done

        Returns:
            {list} -- Data arranged into sequences of variable length as list
        """
        sequences = []
        max_length = 1
        start_index = 0
        for done_index in episode_done_indices:
            episode = data[start_index : done_index + 1]
            if self.sequence_length > 0:
                for seq_start in range(0, len(episode), self.sequence_length):
                    seq = episode[seq_start : seq_start + self.sequence_length]
                    sequences.append(seq)
                    max_length = max(max_length, len(seq))
            else:
                sequences.append(episode)
                max_length = max(max_length, len(episode))
            start_index = done_index + 1
        return sequences, max_length

    def recurrent_mini_batch_generator(self):
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // self.n_mini_batches
        num_sequences_per_batch = (
            [num_sequences_per_batch] * self.n_mini_batches
        )  # Arrange a list that determines the sequence count for each mini batch
        remainder = self.num_sequences % self.n_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += (
                1  # Add the remainder if the sequence count and the number of mini batches do not share a common divider
            )
        # Prepare indices, but only shuffle the sequence indices and not the entire batch to ensure that sequences are maintained as a whole.
        indices = torch.arange(0, self.num_sequences * self.actual_sequence_length).reshape(
            self.num_sequences, self.actual_sequence_length
        )
        sequence_indices = torch.randperm(self.num_sequences)

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_padded_indices = indices[sequence_indices[start:end]].reshape(-1)
            # Unpadded and flat indices are used to sample unpadded training data
            mini_batch_unpadded_indices = self.flat_sequence_indices[
                sequence_indices[start:end].tolist()
            ]
            mini_batch_unpadded_indices = [
                item for sublist in mini_batch_unpadded_indices for item in sublist
            ]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "hxs" or key == "cxs":
                    # Select recurrent cell states of sequence starts
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
                elif key == "log_probs" or "advantages" in key or key == "values":
                    # Select unpadded data
                    mini_batch[key] = value[mini_batch_unpadded_indices].to(self.device)
                else:
                    # Select padded data
                    mini_batch[key] = value[mini_batch_padded_indices].to(self.device)
            start = end
            yield mini_batch

    @torch.no_grad()
    def calc_advantages(self, last_value: torch.tensor, gamma: float, td_lambda: float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.tensor} -- Value of the last agent's state
            gamma {float} -- Discount factor
            td_lambda {float} -- GAE regularization parameter
        """
        mask = torch.logical_not(torch.from_numpy(self.dones))
        rewards = torch.from_numpy(self.rewards)
        values = self.values
        last_value = last_value.squeeze().cpu()
        last_advantage = torch.zeros_like(last_value)
        for t in reversed(range(self.worker_steps)):
            if not mask[t]:
                last_value = torch.zeros_like(last_value)
                last_advantage = torch.zeros_like(last_advantage)
            delta = rewards[t] + gamma * last_value - values[t]
            last_advantage = delta + gamma * td_lambda * last_advantage
            self.advantages[t] = last_advantage
            last_value = values[t]


class ActorCriticModel(nn.Module):
    def __init__(self, hidden_size, layer_type, observation_space, action_space_shape):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_type = layer_type

        # Observation encoder
        self.encoder_type = "simple_cnn"
        if self.encoder_type == "simple_cnn":
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        elif self.encoder_type == "ae":
            self.ae = AutoencoderTiny.from_pretrained("madebyollin/taesd", cache_dir="./cache")

        # Compute output size of convolutional layers
        in_features_next_layer = self.get_conv_output(observation_space.shape)
        print(f"{in_features_next_layer=}")

        self.lin_hidden_in = nn.Linear(in_features_next_layer, self.hidden_size)

        # Recurrent layer (GRU or LSTM)
        if self.layer_type == "gru":
            self.recurrent_layer = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.layer_type == "lstm":
            self.recurrent_layer = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

        # Hidden layer
        self.lin_hidden_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        assert len(action_space_shape) == 1
        self.policy = nn.Linear(in_features=self.hidden_size, out_features=action_space_shape[0])

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)

        # Apply weight initialization to all modules
        self.apply(self._init_weights)

    def forward(
        self,
        obs: torch.tensor,
        recurrent_cell: torch.tensor,
        sequence_length: int = 1,
    ):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations  (B, 3, H, W)
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        B = h.shape[0]
        if self.encoder_type == "simple_cnn":
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
        elif self.encoder_type == "ae":
            h = self.ae.encode(h).latents

        h = h.flatten(start_dim=1)
        h = F.relu(self.lin_hidden_in(h))

        # Forward recurrent layer (GRU or LSTM) first, then hidden layer
        # Reshape the to be fed data to batch_size, sequence_length, data
        B, D = h.shape
        h = h.reshape((B // sequence_length, sequence_length, D))

        # Forward recurrent layer
        h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

        # Reshape to the original tensor size
        B, T, D = h.shape
        h = h.reshape(B * T, D)

        # Feed hidden layer after recurrent layer
        h = F.relu(self.lin_hidden_out(h))
        memory_out = recurrent_cell

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = Categorical(logits=self.policy(h_policy))

        return pi, value, memory_out

    def _init_weights(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def get_conv_output(self, shape: tuple) -> int:
        o = torch.zeros(1, *shape)
        if self.encoder_type == "simple_cnn":
            o = self.conv1(o)
            o = self.conv2(o)
            o = self.conv3(o)
        elif self.encoder_type == "ae":
            o = self.ae.encode(o).latents
        return int(np.prod(o.size()))

    def init_recurrent_cell_states(self, num_sequences: int, device: torch.device) -> tuple:
        hxs = torch.zeros(
            (num_sequences),
            self.hidden_size,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        cxs = None
        if self.layer_type == "lstm":
            cxs = torch.zeros(
                (num_sequences),
                self.hidden_size,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
        return hxs, cxs


class RecurrentPpoAgent:
    def __init__(self, args, observation_space, action_space) -> None:
        # Set variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_space = observation_space
        self.action_space_shape = (action_space.n,)

        self.worker_steps = args.buffer_capacity
        self.layer_type = "gru"
        hidden_size = 256
        sequence_length = args.seq_len

        # Init buffer
        self.buffer = Buffer(
            self.worker_steps,
            hidden_size,
            self.layer_type,
            sequence_length,
            self.observation_space,
            self.action_space_shape,
            self.device,
        )
        self.buffer_index = 0

        # Init model
        self.network = ActorCriticModel(
            hidden_size,
            self.layer_type,
            self.observation_space,
            self.action_space_shape,
        ).to(self.device)
        self.network.train()
        self.optimizer = optim.AdamW(self.network.parameters(), lr=2.0e-4)

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.network.init_recurrent_cell_states(1, self.device)
        if self.layer_type == "gru":
            self.recurrent_cell = hxs
        elif self.layer_type == "lstm":
            self.recurrent_cell = (hxs, cxs)

        self.epoch = 0

    def initialize_for_episode(self) -> None:
        self._train(obs=None)

    @torch.no_grad()
    def select_action(self, global_step, obs, reward) -> tuple[np.ndarray, dict]:
        t = self.buffer_index
        self.buffer_index += 1

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        self.buffer.obs[t] = obs_tensor.cpu()

        current_cell = self.recurrent_cell
        if self.layer_type == "gru":
            self.buffer.hxs[t] = current_cell.squeeze(0).squeeze(0).detach().cpu()
        elif self.layer_type == "lstm":
            self.buffer.hxs[t] = current_cell[0].squeeze(0).squeeze(0).detach().cpu()
            self.buffer.cxs[t] = current_cell[1].squeeze(0).squeeze(0).detach().cpu()

        # Forward the model to retrieve the policy, the states' value and the recurrent cell states
        policy, value, self.recurrent_cell = self.network(
            obs_tensor.unsqueeze(0).to(self.device), current_cell
        )
        self.buffer.values[t] = value.squeeze(0).detach().cpu()

        # Sample actions from each individual policy branch
        actions = []
        log_probs = []
        action = policy.sample()
        actions.append(action)
        log_probs.append(policy.log_prob(action))
        action_tensor = torch.stack(actions, dim=1).detach()
        log_prob_tensor = torch.stack(log_probs, dim=1).detach()
        self.buffer.actions[t] = action_tensor.squeeze(0).cpu().long()
        self.buffer.log_probs[t] = log_prob_tensor.squeeze(0).cpu()

        action_info = {
            "a_logp": log_prob_tensor.cpu().numpy(),
            "value": value.cpu().numpy(),
            "reward": reward,
            "normed_reward": reward,
        }

        return action, action_info

    def step(self, global_step, obs, reward, termination, truncation) -> tuple[np.ndarray, dict]:
        info_dict = {}

        # store
        if self.buffer_index < self.worker_steps:
            done = termination or truncation
            self.buffer.rewards[self.buffer_index] = reward
            self.buffer.dones[self.buffer_index] = done

        # train
        train_info = self._train(obs)
        info_dict.update(train_info)

        # act
        action, action_info = self.select_action(global_step, obs, reward)
        info_dict.update(action_info)

        return action, info_dict

    def _train(self, obs) -> dict:
        if self.buffer_index < self.worker_steps:
            return {}
        self.buffer_index = 0

        # Calculate advantages
        if obs is None:
            last_value = torch.zeros((1, 1), dtype=torch.float32).to(self.device)
        else:
            last_obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, last_value, _ = self.network(last_obs_tensor, self.recurrent_cell)
        self.buffer.calc_advantages(last_value, gamma=0.99, td_lambda=0.95)

        self.buffer.prepare_batch_dict()

        self.epoch += 1
        train_info = {"epoch": self.epoch}
        for _ in range(4):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                self._train_mini_batch(mini_batch)
        return train_info

    def _train_mini_batch(self, samples: dict) -> list:
        """Uses one mini batch to optimize the model.

        Returns:
            {list} -- list of training statistics (e.g. loss)
        """
        beta = 0.02
        clip_range = 0.2

        # Retrieve sampled recurrent cell states to feed the model
        if self.layer_type == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.layer_type == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Forward model
        policy, value, _ = self.network(
            samples["obs"], recurrent_cell, self.buffer.actual_sequence_length
        )

        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        log_probs.append(policy.log_prob(samples["actions"][:, 0]))
        entropies.append(policy.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Remove paddings
        value = value[samples["loss_mask"]]
        log_probs = log_probs[samples["loss_mask"]]
        entropies = entropies[samples["loss_mask"]]

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (
            samples["advantages"].std() + 1e-8
        )
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(
            1, len(self.action_space_shape)
        )  # Repeat is necessary for multi-discrete action spaces
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(
            min=-clip_range, max=clip_range
        )
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - 0.25 * vf_loss + beta * entropy_bonus)

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        return [
            policy_loss.cpu().data.numpy(),
            vf_loss.cpu().data.numpy(),
            loss.cpu().data.numpy(),
            entropy_bonus.cpu().data.numpy(),
        ]
