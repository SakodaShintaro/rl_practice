import argparse

import numpy as np
import torch
from hl_gauss_pytorch import HLGaussLoss
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from .image_processor import ImageProcessor
from .vlm_backbone import ACTION_PROMPT, parse_action_text


class VLMActorCriticWithStateValue(nn.Module):
    """VLM-based Actor-Critic with text action output and state value estimation.

    Actor: Generates action as text via autoregressive decoding.
    Critic: Uses the final hidden state after action token generation to compute V(s).
    """

    def __init__(
        self,
        *,
        observation_space_shape: tuple[int, ...],
        action_dim: int,
        seq_len: int,
        model: nn.Module,
        processor: AutoProcessor,
        use_lora: bool,
        task_prompt: str,
        value_hidden_dim: int,
        target_layer_idx: int,
        max_new_tokens: int,
        num_bins: int,
        value_min: float,
        value_max: float,
        gamma: float,
        image_processor_type: str,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.task_prompt = task_prompt
        self.target_layer_idx = target_layer_idx
        self.max_new_tokens = max_new_tokens
        self.num_bins = num_bins
        self.value_min = value_min
        self.value_max = value_max
        self.gamma = gamma

        self.model = model
        device = next(model.parameters()).device
        self.device = device
        self.processor = processor

        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=[
                    "down_proj",
                    "o_proj",
                    "k_proj",
                    "q_proj",
                    "gate_proj",
                    "up_proj",
                    "v_proj",
                ],
                use_dora=True,
                init_lora_weights="gaussian",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        hidden_size = int(self.model.config.text_config.hidden_size)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, num_bins),
        ).to(device)

        if self.num_bins > 1:
            self.hl_gauss_loss = HLGaussLoss(
                min_value=self.value_min,
                max_value=self.value_max,
                num_bins=self.num_bins,
                clamp_to_range=True,
            ).to(device)

        self.image_processor = ImageProcessor(observation_space_shape, image_processor_type).to(
            device
        )
        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def _build_messages(
        self,
        images: torch.Tensor,
        rewards: torch.Tensor,
    ) -> list[list[dict]]:
        batch_size, seq_len = images.shape[:2]
        messages = []
        for b in range(batch_size):
            content: list[dict] = []
            if self.task_prompt:
                content.append({"type": "text", "text": self.task_prompt})
            for t in range(seq_len):
                img_tensor = images[b, t].to(torch.float32)
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                content.append({"type": "image", "image": Image.fromarray(img_np)})
                reward_text = f"reward {float(rewards[b, t, 0]):.2f}"
                content.append({"type": "text", "text": reward_text})
            messages.append([{"role": "user", "content": content}])
        return messages

    def _prepare_inputs(self, messages: list[list[dict]]) -> dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos = None
            video_metadata = None

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in inputs.items()
        }
        return inputs

    def _generate_with_hidden_states(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[str, torch.Tensor, torch.Tensor, list[int]]:
        """Generate action text and return hidden state, log prob, and token ids.

        Returns:
            action_text: Generated action text
            state_hidden: Hidden state from first forward pass (before action generation)
            total_log_prob: Sum of log probabilities for all generated tokens
            generated_ids: List of generated token IDs
        """
        # First, get hidden state from prompt processing
        outputs = self.model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        state_hidden = hidden_states[self.target_layer_idx][:, -1, :].to(torch.float32)

        # Generate action text using model.generate()
        pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, input_len:]
        generated_ids = new_tokens[0].tolist()

        action_text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # Compute log prob by re-encoding with teacher forcing
        # For now, return 0 as placeholder (actual log prob computed in compute_loss)
        total_log_prob = torch.tensor(0.0, device=self.device)

        return action_text, state_hidden, total_log_prob, generated_ids

    def _encode_for_value(
        self,
        images: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[str, torch.Tensor, torch.Tensor, list[int]]:
        """Encode observation and generate action, returning hidden state and log prob."""
        messages = self._build_messages(images, rewards)
        inputs = self._prepare_inputs(messages)
        action_text, hidden, log_prob, token_ids = self._generate_with_hidden_states(inputs)
        return action_text, hidden, log_prob, token_ids

    def _convert_3d_to_2d_action(self, action_3d: np.ndarray) -> np.ndarray:
        """Convert 3D action [steer, gas, braking] to 2D action [steer, gas_or_brake].

        gas_or_brake = gas - braking, clamped to [-1, 1]
        """
        steer = action_3d[0]
        gas = action_3d[1]
        braking = action_3d[2]
        gas_or_brake = np.clip(gas - braking, -1.0, 1.0)
        return np.array([steer, gas_or_brake], dtype=np.float32)

    def forward(
        self,
        s_seq: torch.Tensor,
        obs_z_seq: torch.Tensor,
        a_seq: torch.Tensor,
        r_seq: torch.Tensor,
        rnn_state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:
        """Forward pass: generate action text and compute state value."""
        action_text, hidden, log_prob, _ = self._encode_for_value(s_seq, r_seq)
        action_array_3d = parse_action_text(action_text)
        action_array = self._convert_3d_to_2d_action(action_array_3d)
        action_tensor = torch.from_numpy(action_array).unsqueeze(0).to(s_seq.device)

        value_logits = self.value_head(hidden)
        if self.num_bins > 1:
            value = self.hl_gauss_loss(value_logits).unsqueeze(-1)
        else:
            value = value_logits

        # log_prob is already a scalar (sum of token log probs)
        a_logp = log_prob.unsqueeze(0).unsqueeze(-1)
        entropy = torch.zeros((s_seq.size(0), 1), device=s_seq.device)

        return {
            "action": action_tensor,
            "a_logp": a_logp,
            "entropy": entropy,
            "value": value,
            "x": hidden,
            "rnn_state": rnn_state,
            "action_text": action_text,
        }

    def _compute_value_from_state(
        self,
        images: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute value from state hidden (before action generation).

        Processes the prompt and returns the hidden state from the first forward pass,
        which is the same information available for action generation.
        """
        messages = self._build_messages(images, rewards)
        inputs = self._prepare_inputs(messages)

        outputs = self.model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        state_hidden = hidden_states[self.target_layer_idx][:, -1, :].to(torch.float32)

        value_logits = self.value_head(state_hidden)
        if self.num_bins > 1:
            value = self.hl_gauss_loss(value_logits).unsqueeze(-1)
        else:
            value = value_logits

        return value, state_hidden, inputs

    def _action_to_text(self, action: torch.Tensor) -> str:
        """Convert 2D action tensor to canonical text representation.

        2D action: [steer, gas_or_brake] -> 3D text: steering, gas, braking
        """
        steering = action[0].item()
        gas_or_brake = action[1].item()
        gas = max(gas_or_brake, 0.0)
        braking = max(-gas_or_brake, 0.0)
        return f"Action: steering={steering:.2f}, gas={gas:.2f}, braking={braking:.2f}"

    def _compute_log_prob_for_actions_batch(
        self,
        inputs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a batch of actions using teacher forcing.

        Args:
            inputs: Prepared model inputs (already batched)
            actions: Action tensors of shape (B, action_dim)

        Returns:
            Log probabilities of shape (B,)
        """
        batch_size = actions.size(0)

        # Convert all actions to text and tokenize
        action_texts = [self._action_to_text(actions[b]) for b in range(batch_size)]
        tokenized = self.processor.tokenizer(
            action_texts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        target_ids = tokenized["input_ids"].to(self.device)
        target_mask = tokenized["attention_mask"].to(self.device).float()
        action_len = target_ids.size(1)

        # Concatenate prompt input_ids with action token ids
        # input_ids: (B, prompt_len), target_ids: (B, action_len)
        combined_input_ids = torch.cat([inputs["input_ids"], target_ids], dim=1)
        combined_attention_mask = torch.cat(
            [inputs["attention_mask"], tokenized["attention_mask"].to(self.device)], dim=1
        )

        # Single forward pass with all tokens
        outputs = self.model.forward(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_hidden_states=False,
            return_dict=True,
        )

        # logits: (B, prompt_len + action_len, vocab_size)
        # We need logits at positions [prompt_len-1, prompt_len, ..., prompt_len+action_len-2]
        # These predict tokens at positions [prompt_len, prompt_len+1, ..., prompt_len+action_len-1]
        prompt_len = inputs["input_ids"].size(1)
        relevant_logits = outputs.logits[:, prompt_len - 1 : prompt_len + action_len - 1, :]

        # Compute log probs
        log_prob_dist = F.log_softmax(relevant_logits, dim=-1)

        # Gather log probs for target tokens: (B, action_len)
        token_log_probs = log_prob_dist.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        # Sum log probs, masking padding tokens
        batch_log_probs = (token_log_probs * target_mask).sum(dim=1)

        return batch_log_probs

    clip_param_policy = 0.2
    clip_param_value = 0.2

    def compute_loss(
        self,
        data,
        curr_target_v: torch.Tensor,
        curr_adv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict, dict]:
        """Compute PPO actor loss and critic loss."""
        obs_curr = data.observations[:, :-1]
        rewards_curr = data.rewards[:, :-1]
        actions_curr = data.actions[:, -1]  # (B, action_dim)
        old_log_prob = data.log_probs[:, -1].view(-1)

        # Compute value and hidden states for batch
        value, hidden, inputs = self._compute_value_from_state(obs_curr, rewards_curr)

        # Compute new log prob for actions in batch
        new_log_prob = self._compute_log_prob_for_actions_batch(inputs, actions_curr)

        # PPO actor loss
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * curr_adv.view(-1)
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_param_policy, 1.0 + self.clip_param_policy
        ) * curr_adv.view(-1)
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss
        if self.num_bins > 1:
            value_loss = self.hl_gauss_loss(self.value_head(hidden), curr_target_v.view(-1))
        else:
            value_loss = F.mse_loss(value.view(-1), curr_target_v.view(-1))

        total_loss = actor_loss + value_loss

        activations_dict = {
            "critic": hidden,
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "critic_loss": value_loss.item(),
        }

        return total_loss, activations_dict, info_dict

    @torch.no_grad()
    def compute_target_value(self, data) -> torch.Tensor:
        """Compute target value for TD learning."""
        obs_next = data.observations[:, 1:]
        rewards_next = data.rewards[:, 1:]
        dones_next = data.dones[:, -1].flatten()

        _, hidden_next, _ = self._compute_value_from_state(obs_next, rewards_next)
        value_logits_next = self.value_head(hidden_next)
        if self.num_bins > 1:
            next_value = self.hl_gauss_loss(value_logits_next)
        else:
            next_value = value_logits_next.view(-1)

        curr_reward = data.rewards[:, -1].flatten()
        curr_continue = 1 - dones_next

        return curr_reward + curr_continue * self.gamma * next_value


def create_vlm_actor_critic_network(
    observation_space_shape: tuple[int, ...],
    action_space_shape: tuple[int, ...],
    args: argparse.Namespace,
) -> VLMActorCriticWithStateValue:
    """Factory function to create VLMActorCriticWithStateValue from args."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

    model = AutoModelForImageTextToText.from_pretrained(
        args.vlm_model_id,
        dtype=torch.bfloat16,
        _attn_implementation=attn_impl,
        cache_dir="./cache",
        device_map=device,
    )

    processor = AutoProcessor.from_pretrained(args.vlm_model_id, cache_dir="./cache")

    return VLMActorCriticWithStateValue(
        observation_space_shape=observation_space_shape,
        action_dim=action_space_shape[0],
        seq_len=args.seq_len,
        model=model,
        processor=processor,
        use_lora=bool(args.use_lora),
        task_prompt=ACTION_PROMPT,
        value_hidden_dim=args.critic_hidden_dim,
        target_layer_idx=args.target_layer_idx,
        max_new_tokens=512,
        num_bins=args.num_bins,
        value_min=-args.value_range,
        value_max=+args.value_range,
        gamma=args.gamma,
        image_processor_type=args.image_processor_type,
    )
