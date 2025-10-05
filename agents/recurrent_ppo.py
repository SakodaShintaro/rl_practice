# Derived from https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt
import numpy as np
import torch
from gymnasium import spaces
from torch import optim

from networks.actor_critic_with_state_value import Network1


class Buffer:
    """The buffer stores and prepares the training data. It supports recurrent policies."""

    def __init__(
        self,
        worker_steps: int,
        hidden_size: int,
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
        self.sequence_length = sequence_length
        self.true_sequence_length = 0

        # Initialize the buffer's data storage for a single environment
        self.rewards = np.zeros(self.worker_steps, dtype=np.float32)
        self.actions = torch.zeros((self.worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros(self.worker_steps, dtype=bool)
        self.obs = torch.zeros((self.worker_steps,) + observation_space.shape)
        self.hxs = torch.zeros((self.worker_steps, hidden_size))
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
            if key in ("hxs"):
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
            if key in ("hxs"):
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
        n_mini_batches = min(self.n_mini_batches, self.num_sequences)
        num_sequences_per_batch = self.num_sequences // n_mini_batches
        num_sequences_per_batch = [
            num_sequences_per_batch
        ] * n_mini_batches  # Arrange a list that determines the sequence count for each mini batch
        remainder = self.num_sequences % n_mini_batches
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
                if key == "hxs":
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
    def calc_advantages(self, last_value: torch.Tensor, gamma: float, td_lambda: float) -> None:
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {torch.Tensor} -- Value of the last agent's state
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


class RecurrentPpoAgent:
    def __init__(self, args, observation_space, action_space) -> None:
        # Set variables
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.observation_space = observation_space
        self.action_space_shape = action_space.shape

        self.worker_steps = args.buffer_capacity
        hidden_size = 256
        sequence_length = args.seq_len

        # Init buffer
        self.buffer = Buffer(
            self.worker_steps,
            hidden_size,
            sequence_length,
            self.observation_space,
            self.action_space_shape,
            self.device,
        )
        self.buffer_index = 0

        # Init model
        self.network = Network1(
            hidden_size,
            self.observation_space,
            self.action_space_shape,
        ).to(self.device)
        self.network.train()
        self.optimizer = optim.AdamW(self.network.parameters(), lr=args.learning_rate)

        # Setup initial recurrent cell states
        self.recurrent_cell = self.network.init_state().to(self.device)

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
        self.buffer.hxs[t] = current_cell.squeeze(0).squeeze(0).detach().cpu()

        # Forward the model to retrieve the policy, the states' value and the recurrent cell states
        obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, C, H, W)
        output_dict = self.network(obs_tensor, current_cell)
        action = output_dict["action"]
        log_prob = output_dict["a_logp"]
        value = output_dict["value"]
        self.recurrent_cell = output_dict["recurrent_cell"]

        self.buffer.actions[t] = action.squeeze(0).cpu().long()
        self.buffer.log_probs[t] = log_prob.squeeze(0).cpu()
        self.buffer.values[t] = value.squeeze(0).detach().cpu()

        action_info = {
            "a_logp": log_prob.cpu().numpy(),
            "value": value.cpu().numpy(),
            "reward": reward,
            "normed_reward": reward,
        }

        # Convert discrete action to continuous format for the environment
        continuous_action = np.zeros(self.action_space_shape, dtype=np.float32)
        discrete_action = action.cpu().numpy()
        continuous_action[discrete_action] = 1.0
        return continuous_action, action_info

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
            last_obs_tensor = (
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            )  # (1, 1, C, H, W)
            output_dict = self.network(last_obs_tensor, self.recurrent_cell)
            last_value = output_dict["value"]
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
        recurrent_cell = samples["hxs"].unsqueeze(0)

        # Forward model
        _, B, _ = recurrent_cell.shape
        obs = (
            samples["obs"]
            .to(self.device)
            .reshape(B, self.buffer.actual_sequence_length, *self.observation_space.shape)
        )
        output_dict = self.network(obs, recurrent_cell, samples["actions"][:, 0].view(B, -1))
        mask = samples["loss_mask"]
        value = output_dict["value"].flatten()[mask].unsqueeze(1)
        log_probs = output_dict["a_logp"].flatten()[mask].unsqueeze(1)
        entropies = output_dict["entropy"].flatten()[mask].unsqueeze(1)

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
