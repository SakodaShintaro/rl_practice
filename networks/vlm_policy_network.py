import numpy as np
import torch
from hl_gauss_pytorch import HLGaussLoss
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor

from .image_processor import ImageProcessor


class VLMPolicyNetwork(nn.Module):
    """VLM-based policy (and optional action value) network."""

    def __init__(
        self,
        *,
        action_dim: int,
        seq_len: int,
        action_horizon: int,
        observation_space_shape: tuple[int, ...],
        image_processor_type: str,
        target_layer_idx: int,
        model: nn.Module,
        processor: AutoProcessor,
        use_lora: bool,
        task_prompt: str,
        action_hidden_dim: int,
        value_hidden_dim: int,
        value_bins: int,
        value_min: float,
        value_max: float,
        euler_steps: int,
        gamma: float,
        dacer_loss_weight: float,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.action_horizon = action_horizon
        self.target_layer_idx = target_layer_idx
        self.task_prompt = task_prompt
        self.value_bins = value_bins
        self.value_min = value_min
        self.value_max = value_max
        self.euler_steps = euler_steps
        self.gamma = gamma
        self.dacer_loss_weight = dacer_loss_weight
        if self.value_bins <= 1:
            raise ValueError("value_bins must be > 1 for HLGauss-based value modeling.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model
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
        self.image_processor = ImageProcessor(
            observation_space_shape, processor_type=image_processor_type
        ).to(self.device)

        hidden_size = int(self.model.config.text_config.hidden_size)
        self.action_in_proj = nn.Linear(action_dim, hidden_size)
        self.action_value_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, value_bins),
        )
        self.num_bins = value_bins
        self.hl_gauss_loss = HLGaussLoss(
            min_value=self.value_min,
            max_value=self.value_max,
            num_bins=self.num_bins,
            clamp_to_range=True,
        )

        self._dummy_state = torch.zeros(1, 1, 1)

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def _prepare_prefix_inputs(
        self,
        images: torch.Tensor,  # (B, T, C, H, W)
        text_seq: list[list[str]],  # (B, T)
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = images.shape[:2]
        messages = []
        for b in range(batch_size):
            content: list[dict[str, object]] = []
            if self.task_prompt:
                content.append({"type": "text", "text": self.task_prompt})
            for t in range(seq_len):
                img_tensor = images[b, t].to(torch.float32)
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                content.append({"type": "image", "image": Image.fromarray(img_np)})
                step_text = text_seq[b][t].strip()
                if step_text:
                    content.append({"type": "text", "text": step_text})
            messages.append([{"role": "user", "content": content}])

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
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in inputs.items()
        }
        return inputs

    def _encode(
        self,
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        text_seq: list[list[str]],  # (B, T)
        action: torch.Tensor,  # (B, H, action_dim)
        compute_action_value: bool,
    ) -> torch.Tensor:
        batch_size, seq_len = s_seq.shape[:2]
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")

        if len(text_seq) != batch_size:
            raise ValueError("text_seq batch size does not match images batch size.")
        if any(len(row) != seq_len for row in text_seq):
            raise ValueError("text_seq sequence length does not match images sequence length.")

        # Prefix: image/text tokens from the processor, Suffix: embedded action/action_value tokens.
        prefix_inputs = self._prepare_prefix_inputs(s_seq, text_seq)
        input_ids = prefix_inputs["input_ids"]
        attention_mask = prefix_inputs["attention_mask"]
        model_dtype = self.model.get_input_embeddings().weight.dtype
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(model_dtype)

        # Suffix
        suffix_embs = self.action_in_proj(action).to(model_dtype)
        if compute_action_value:
            action_value_token = self.action_value_token.to(model_dtype).expand(
                action.size(0), 1, -1
            )
            suffix_embs = torch.cat([suffix_embs, action_value_token], dim=1)
        suffix_mask = torch.ones(
            (attention_mask.size(0), suffix_embs.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        # Concatenate prefix and suffix
        inputs_embeds = torch.cat([inputs_embeds, suffix_embs], dim=1)
        attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)

        # Forward pass
        base_model = self.model.model if hasattr(self.model, "model") else self.model
        outputs = base_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=prefix_inputs["pixel_values"],
            image_grid_thw=prefix_inputs["image_grid_thw"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = (
            outputs["hidden_states"] if isinstance(outputs, dict) else outputs.hidden_states
        )
        hidden = hidden_states[self.target_layer_idx]

        # Return the relevant hidden states
        if compute_action_value:
            return hidden[:, -1, :].to(torch.float32)
        return hidden[:, -suffix_embs.size(1) :, :].to(torch.float32)

    def _compute_actor_loss(
        self,
        image: torch.Tensor,  # (B, T, C, H, W)
        text: list[list[str]],  # (B, T)
        action: torch.Tensor,  # (B, H, action_dim)
    ) -> tuple[torch.Tensor, dict, dict]:
        batch_size = action.size(0)
        device = action.device

        # Policy rollout (Euler) to get pi
        pi = torch.zeros_like(action)
        dt = 1.0 / float(self.euler_steps)
        for _ in range(self.euler_steps):
            pi_hidden = self._encode(image, text, pi, False)
            velocity = self.action_head(pi_hidden)
            pi = pi + dt * velocity

        # Actor loss (-Q)
        for param in self.value_head.parameters():
            param.requires_grad_(False)
        critic_pi_hidden = self._encode(image, text, pi, True)
        critic_pi_logits = self.value_head(critic_pi_hidden)
        critic_pi = self.hl_gauss_loss(critic_pi_logits).unsqueeze(-1)
        actor_loss = -critic_pi.mean()
        for param in self.value_head.parameters():
            param.requires_grad_(True)

        # DACER-like loss (adapted to action horizon)
        actions = pi.clone().detach()
        actions.requires_grad = True
        eps = 1e-4
        t = (torch.rand((batch_size, 1, 1), device=device)) * (1 - eps) + eps
        c = 0.4
        d = -1.8
        w_t = torch.exp(c * t + d)

        def calc_target(actions_local):
            q_hidden = self._encode(image, text, actions_local, True)
            q_logits = self.value_head(q_hidden)
            q_values = self.hl_gauss_loss(q_logits).unsqueeze(-1)
            q_grad = torch.autograd.grad(
                outputs=q_values.sum(),
                inputs=actions_local,
                create_graph=True,
            )[0]
            with torch.no_grad():
                target = (1 - t) / t * q_grad + 1 / t * actions_local
                target = target / (target.norm(dim=2, keepdim=True) + 1e-8)
                return w_t * target

        target = calc_target(actions)
        noise = torch.randn_like(actions).clamp(-3.0, 3.0)
        a_t = (1.0 - t) * noise + t * actions
        actor_hidden = self._encode(image, text, a_t, False)
        v = self.action_head(actor_hidden)
        dacer_loss = F.mse_loss(v, target)

        total_actor_loss = actor_loss + dacer_loss * self.dacer_loss_weight

        activations_dict = {
            "actor": actor_hidden,
            "critic": critic_pi_hidden,
        }

        info_dict = {
            "actor_loss": actor_loss.item(),
            "dacer_loss": dacer_loss.item(),
        }

        return total_actor_loss, activations_dict, info_dict

    def _compute_value_loss(
        self,
        image: torch.Tensor,  # (B, T, C, H, W)
        text: list[list[str]],  # (B, T)
        action: torch.Tensor,  # (B, H, action_dim)
        target_q: torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        value_hidden = self._encode(image, text, action, True)
        value_output = self.value_head(value_hidden)
        return self.hl_gauss_loss(value_output, target_q.view(-1))

    @torch.inference_mode()
    def _infer_action(
        self,
        image: torch.Tensor,  # (B, T, C, H, W)
        text: list[list[str]],  # (B, T)
    ) -> torch.Tensor:
        self.model.eval()
        batch_size = image.size(0)
        device = image.device
        action = torch.zeros((batch_size, self.action_horizon, self.action_dim), device=device)
        dt = 1.0 / float(self.euler_steps)
        for _ in range(self.euler_steps):
            action_hidden = self._encode(image, text, action, False)
            velocity = self.action_head(action_hidden)
            action = action + dt * velocity
        return torch.tanh(action)

    @torch.inference_mode()
    def _infer_action_value(
        self,
        image: torch.Tensor,  # (B, T, C, H, W)
        text: list[list[str]],  # (B, T)
        action: torch.Tensor,  # (B, H, action_dim)
    ) -> torch.Tensor:
        self.model.eval()
        value_hidden = self._encode(image, text, action, True)
        value_output = self.value_head(value_hidden)
        return self.hl_gauss_loss(value_output).unsqueeze(-1)

    @torch.inference_mode()
    def infer(
        self,
        s_seq: torch.Tensor,  # (B, T, C, H, W)
        obs_z_seq: torch.Tensor,  # (B, T, C', H', W') - unused
        a_seq: torch.Tensor,  # (B, T, action_dim)
        r_seq: torch.Tensor,  # (B, T, 1)
        rnn_state: torch.Tensor,
    ) -> dict:
        text_seq = self._build_text_seq(r_seq)
        action_seq = self._infer_action(s_seq, text_seq)
        value = self._infer_action_value(s_seq, text_seq, action_seq)
        action_hidden = self._encode(s_seq, text_seq, action_seq, False)
        x = action_hidden[:, -1, :]
        a_logp = torch.zeros((s_seq.size(0), 1), device=s_seq.device)
        return {
            "action": action_seq[:, -1, :],
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "rnn_state": rnn_state,
        }

    def compute_loss(self, data, target_value) -> tuple[torch.Tensor, dict, dict]:
        obs_curr = data.observations[:, :-1]
        actions_curr = self._slice_actions(data.actions[:, :-1])
        rewards_curr = data.rewards[:, :-1]
        text_curr = self._build_text_seq(rewards_curr)
        actor_loss, actor_activations, actor_info = self._compute_actor_loss(
            obs_curr, text_curr, actions_curr
        )
        critic_loss = self._compute_value_loss(obs_curr, text_curr, actions_curr, target_value)
        total_loss = critic_loss + actor_loss
        activations_dict = {
            **actor_activations,
        }
        info_dict = {
            **actor_info,
            "critic_loss": critic_loss.item(),
            "seq_loss": 0.0,
        }
        return total_loss, activations_dict, info_dict

    @torch.no_grad()
    def compute_target_value(self, data) -> torch.Tensor:
        obs_next = data.observations[:, 1:]
        rewards_next = data.rewards[:, 1:]
        dones_next = data.dones[:, -1].flatten()
        text_next = self._build_text_seq(rewards_next)
        action_next = self._infer_action(obs_next, text_next)
        next_value = self._infer_action_value(obs_next, text_next, action_next).view(-1)
        curr_reward = data.rewards[:, -1].flatten()
        curr_continue = 1 - dones_next
        return curr_reward + curr_continue * self.gamma * next_value

    def _build_text_seq(self, rewards: torch.Tensor) -> list[list[str]]:
        rewards_cpu = rewards.detach().cpu().numpy()
        text_seq = []
        for b in range(rewards_cpu.shape[0]):
            row = []
            for t in range(rewards_cpu.shape[1]):
                row.append(f"reward {float(rewards_cpu[b, t, 0]):.3f}")
            text_seq.append(row)
        return text_seq

    def _slice_actions(self, actions: torch.Tensor) -> torch.Tensor:
        horizon = self.action_horizon
        if actions.size(1) >= horizon:
            return actions[:, -horizon:, :]
        pad = torch.zeros(
            (actions.size(0), horizon - actions.size(1), actions.size(2)),
            device=actions.device,
            dtype=actions.dtype,
        )
        return torch.cat([pad, actions], dim=1)
