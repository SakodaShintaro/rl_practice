import re

import numpy as np
import torch
import torchvision.transforms as T
from diffusers.models import AutoencoderTiny
from mamba_ssm.utils.generation import InferenceParams
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .for_mmmamba.modeling_mmMamba_chat import mmMambaChatModel
from .spatial_temporal_transformer import SpatialTemporalTransformer

# Unified action prompt for all VLM encoders
ACTION_PROMPT = (
    "You are an AI driving assistant. Analyze the driving scene from the video frames (from oldest to newest) and provide the next action. "
    "Action space: steering (-1 to +1, where -1 is full left, +1 is full right), "
    "gas (0 to 1), braking (0 to 1). "
    "Stay on the gray road and avoid green areas. "
    "Respond in format: 'Action: steering=X.X, gas=X.X, braking=X.X' where X.X are decimal values."
)


def parse_action_text(action_text: str) -> np.ndarray:
    """Parse action text and extract steering, gas, braking values.

    Args:
        action_text: Text in format 'Action: steering=X.X, gas=X.X, braking=X.X'

    Returns:
        np.ndarray of shape (3,) containing [steering, gas, braking]

    Example:
        >>> parse_action_text("Action: steering=0.5, gas=0.3, braking=0.0")
        array([0.5, 0.3, 0.0])
    """
    # Default values in case parsing fails
    steering, gas, braking = 0.0, 0.0, 0.0

    try:
        # Extract steering value
        steering_match = re.search(r"steering=([+-]?\d*\.?\d+)", action_text)
        if steering_match:
            steering = float(steering_match.group(1))

        # Extract gas value
        gas_match = re.search(r"gas=([+-]?\d*\.?\d+)", action_text)
        if gas_match:
            gas = float(gas_match.group(1))

        # Extract braking value
        braking_match = re.search(r"braking=([+-]?\d*\.?\d+)", action_text)
        if braking_match:
            braking = float(braking_match.group(1))

    except (ValueError, AttributeError):
        # Return default values if parsing fails
        pass

    # Create array and clamp values using numpy.clip
    action_array = np.array([steering, gas, braking], dtype=np.float32)
    action_array[0] = np.clip(action_array[0], -1.0, 1.0)  # steering
    action_array[1] = np.clip(action_array[1], 0.0, 1.0)  # gas
    action_array[2] = np.clip(action_array[2], 0.0, 1.0)  # braking

    return action_array


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class AE(nn.Module):
    def __init__(self, device=None) -> None:
        super().__init__()
        self.ae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", cache_dir="./cache", device_map=device
        )

        # self.ae.apply(init_weights)

        self.output_dim = 576
        self.norm = nn.LayerNorm(self.output_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, str]:
        x = self.ae.encode(x).latents.flatten(1)
        x = self.norm(x)
        return x, ""

    @torch.no_grad()
    def decode(self, x):
        x = x.view(x.size(0), 4, 12, 12)
        return self.ae.decode(x).sample

    def reset_inference_params(self):
        pass


class VLMEncoderBase(nn.Module):
    """Base class for Vision-Language Model encoders"""

    def __init__(self, model_id: str, output_dim: int, device=None):
        super().__init__()

        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.output_dim = output_dim
        self.device = device
        self.max_frames = 100
        self.frame_buffer = []

    def get_stop_tokens(self):
        """Get stop tokens for text generation - to be overridden by subclasses"""
        return [
            self.processor.tokenizer.eos_token_id,
            self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, str]:
        assert images.shape[0] == 1, "Batch size must be 1 for stepwise inference"

        # Convert tensor to PIL Image and add to buffer
        img_tensor = images[0].to(torch.float32)  # (3, H, W) - ensure float32 for PIL conversion
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        self.frame_buffer.append(pil_img)

        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)

        # Create chat template messages with video frames
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": frame} for frame in self.frame_buffer]
                + [{"type": "text", "text": ACTION_PROMPT}],
            }
        ]

        # Prepare model inputs
        model_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in model_inputs.items()
        }

        # Get hidden state from forward pass
        input_len = model_inputs["input_ids"].shape[-1]
        output = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = output["hidden_states"][-1]
        x = hidden[:, input_len - 1].to(torch.float32)

        # Generate action text using logits from forward pass
        action_text = self._generate_action_text(output.logits, model_inputs)

        return x, action_text

    def _generate_action_text(self, logits, model_inputs) -> str:
        """Generate action text from logits without using model.generate()"""
        generated_ids = []
        stop_token_ids = self.get_stop_tokens()
        # Remove None values
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        # Get the next token from the initial logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        input_ids = torch.cat([model_inputs["input_ids"], next_token], dim=-1)

        for _ in range(50):  # max_new_tokens
            if next_token.item() in stop_token_ids:
                break
            generated_ids.append(next_token.item())

            # Manually update attention mask and position ids if they exist
            if "attention_mask" in model_inputs:
                attention_mask = model_inputs["attention_mask"]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device)],
                    dim=-1,
                )
                model_inputs["attention_mask"] = attention_mask

            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=model_inputs.get("attention_mask"),
                output_hidden_states=False,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        action_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return action_text

    def reset_inference_params(self):
        self.frame_buffer = []


class SmolVLMEncoder(VLMEncoderBase):
    def __init__(self, device=None) -> None:
        model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        # model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        # model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        output_dim = 576
        super().__init__(model_id, output_dim, device)


class QwenVLEncoder(VLMEncoderBase):
    def __init__(self, device=None) -> None:
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
        # model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
        output_dim = 1536
        super().__init__(model_id, output_dim, device)


class SequenceEncoderBase(nn.Module):
    """Base class for sequence encoders that wrap image encoders"""

    def __init__(self, image_encoder, seq_len=1):
        super().__init__()
        self.image_encoder = image_encoder
        self.seq_len = seq_len
        self.output_dim = image_encoder.output_dim
        self.frame_buffer = []

    def forward(self, observations):
        """Forward pass for sequence of observations

        Args:
            observations: Tensor of shape (batch_size, seq_len, *image_shape) or (batch_size, *image_shape)

        Returns:
            encoded states and action text from the most recent frame
        """
        if len(observations.shape) == 4:
            # Single observation: (batch_size, C, H, W)
            return self._forward_single(observations)
        else:
            # Sequence of observations: (batch_size, seq_len, C, H, W)
            return self._forward_sequence(observations)

    def _forward_single(self, observation):
        """Process single observation (current behavior)"""
        # Update frame buffer for internal sequence tracking
        if observation.shape[0] == 1:  # batch_size == 1
            self.frame_buffer.append(observation)
            if len(self.frame_buffer) > self.seq_len:
                self.frame_buffer.pop(0)

        # Process using underlying image encoder
        return self.image_encoder.forward(observation)

    def _forward_sequence(self, observations):
        """Process sequence of observations (future implementation)"""
        # For now, just use the most recent frame
        latest_obs = observations[:, -1]  # (batch_size, C, H, W)
        return self._forward_single(latest_obs)

    def reset_inference_params(self):
        """Reset inference parameters"""
        self.frame_buffer = []
        if hasattr(self.image_encoder, "reset_inference_params"):
            self.image_encoder.reset_inference_params()

    def decode(self, x):
        """Decode latent representation back to image"""
        if hasattr(self.image_encoder, "decode"):
            return self.image_encoder.decode(x)
        else:
            raise NotImplementedError("Underlying encoder does not support decoding")


class MMMambaEncoder(nn.Module):
    """
    https://huggingface.co/hustvl/mmMamba-linear/blob/main/modeling_mmMamba_chat.py
    """

    def __init__(self, device=None) -> None:
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = "hustvl/mmMamba-linear"

        self.model = mmMambaChatModel.from_pretrained(
            model_id,
            cache_dir="./cache",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model = self.model.to(device)

        # type(self.model.language_model)=<class 'transformers_modules.hustvl.mmMamba-linear.1198b4cf4cae76d9ea5d50e2c0b9724621d6f4f6.modeling_mmMamba.mmMambaForCausalLM'>
        # print(f"{type(self.model.language_model)=}")  # AutoModel

        # print(f"{self.model.config.embedding_config.img_context_token_id=}")  # 92546=IMG_CONTEXT

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir="./cache", trust_remote_code=True, use_fast=False
        )
        # type(self.tokenizer)=<class 'transformers_modules.hustvl.mmMamba-linear.1198b4cf4cae76d9ea5d50e2c0b9724621d6f4f6.tokenization_internlm2.InternLM2Tokenizer'>
        # print(f"{type(self.tokenizer)=}")

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        self.input_size = 96
        self.image_token_num = (self.input_size // 14 // 2) ** 2

        self.transform = T.Compose(
            [
                T.Resize(
                    (self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)
        self.output_dim = 2048

    def reset_inference_params(self):
        """Reset inference parameters to default values."""
        self.inference_params = InferenceParams(max_seqlen=1024, max_batch_size=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, str]:
        device = image.device
        batch_size = image.shape[0]
        assert batch_size == 1, "Batch size must be 1 for stepwise inference"
        image = self.transform(image).to(device).to(torch.bfloat16)
        messages = [
            {
                "role": "user",
                "content": ACTION_PROMPT
                + " <|im_start|>"
                + "<IMG_CONTEXT>" * self.image_token_num
                + "<|im_end|>",
            }
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = model_inputs["input_ids"].to(device)

        output_ids = []
        stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        # Remove None values
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        outputs = self.model.forward(
            input_ids=input_ids,
            pixel_values=image,
            inference_params=self.inference_params,
            output_hidden_states=True,
        )  # CausalLMOutputWithPast (outputs.keys()=odict_keys(['logits', 'hidden_states']))

        # Get representation from hidden states
        x = outputs["hidden_states"][-1][:, 0]
        x = x.to(torch.float32)

        # Generate action text using multiple tokens
        logits = outputs["logits"]
        last_logit = logits[:, -1, :]  # shape: (batch_size, vocab_size)
        token = torch.argmax(last_logit, dim=-1)  # shape: (batch_size,)

        self.inference_params.seqlen_offset += input_ids.shape[1]
        input_ids = token.unsqueeze(1)  # Start with first generated token

        # Generate more tokens for action text
        for _ in range(50):  # max_new_tokens
            if token.item() in stop_token_ids:
                break
            output_ids.append(token.item())

            outputs = self.model.forward(
                input_ids=input_ids,
                inference_params=self.inference_params,
            )
            logits = outputs["logits"]
            last_logit = logits[:, -1, :]
            token = torch.argmax(last_logit, dim=-1)
            input_ids = token.unsqueeze(1)

        # Decode generated tokens to action text
        action_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return x, action_text


class SequenceAEEncoder(SequenceEncoderBase):
    """Sequence encoder wrapper for AE"""

    def __init__(self, seq_len=1, device=None):
        ae_encoder = AE(device=device)
        super().__init__(ae_encoder, seq_len)


class SequenceSmolVLMEncoder(SequenceEncoderBase):
    """Sequence encoder wrapper for SmolVLM"""

    def __init__(self, seq_len=1, device=None):
        smolvlm_encoder = SmolVLMEncoder(device=device)
        super().__init__(smolvlm_encoder, seq_len)


class SequenceMMMambaEncoder(SequenceEncoderBase):
    """Sequence encoder wrapper for MMMamba"""

    def __init__(self, seq_len=1, device=None):
        mmmamba_encoder = MMMambaEncoder(device=device)
        super().__init__(mmmamba_encoder, seq_len)


class SequenceSTTEncoder(nn.Module):
    """Sequence encoder using SpatialTemporalTransformer"""

    def __init__(self, seq_len, action_dim):
        super().__init__()

        self.seq_len = seq_len
        self.action_dim = action_dim
        self.condition_frames = seq_len

        # Default configuration (reduced for memory efficiency)
        hidden_dim = 128  # Reduced from 256
        img_tokens_size = 1024
        vae_emb_dim = 8
        action_ranges = [(-1.0, 1.0)] * action_dim
        device = "cuda:0"

        self.action_ranges = action_ranges

        action_tokens_size = action_dim
        total_tokens_size = img_tokens_size + action_tokens_size

        token_size_dict = {
            "img_tokens_size": img_tokens_size,
            "action_tokens_size": action_tokens_size,
            "total_tokens_size": total_tokens_size,
        }

        # Use AE encoder for image preprocessing
        self.ae_encoder = AE(device=device)

        # AE encoder outputs [B, 4, 12, 12] -> treat as [B, 144, 4] (144 patches of 4 dims each)
        self.actual_img_tokens_size = 144  # 12 * 12 patches
        actual_vae_emb_dim = 4  # each patch has 4 dimensions

        # Update token size configuration
        token_size_dict = {
            "img_tokens_size": self.actual_img_tokens_size,
            "action_tokens_size": action_dim,
            "total_tokens_size": self.actual_img_tokens_size + action_dim,
        }

        self.stt = SpatialTemporalTransformer(
            block_size=self.actual_img_tokens_size + action_dim,
            n_layer=[3, 2],  # Reduced from [6, 3]
            n_head=4,  # Reduced from 8
            n_embd=hidden_dim,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            n_unmasked=0,
            condition_frames=self.condition_frames,
            latent_size=(12, 12),
            token_size_dict=token_size_dict,
            vae_emb_dim=actual_vae_emb_dim,
            temporal_block=1,
            action_ranges=action_ranges,
        )

        # Add projection layer to match AE encoder output dimension
        stt_output_dim = hidden_dim  # 128 (n_embd from global average)
        ae_output_dim = 576
        self.output_projection = nn.Linear(stt_output_dim, ae_output_dim)
        self.output_dim = ae_output_dim
        self.obs_history = []
        self.action_history = []

    def reset_history(self):
        self.obs_history = []
        self.action_history = []

    def reset_inference_params(self):
        """Reset inference parameters for compatibility with other encoders"""
        self.reset_history()

    def update_history(self, obs, action=None):
        self.obs_history.append(obs)
        if len(self.obs_history) > self.condition_frames + 1:
            self.obs_history.pop(0)

        if action is not None:
            self.action_history.append(action)
            if len(self.action_history) > self.condition_frames + 1:
                self.action_history.pop(0)

    def _prepare_stt_input(self, observations):
        # observations: [B, 3, H, W] - Raw RGB images
        # Encode with AE but preserve spatial structure
        with torch.no_grad():
            latents = self.ae_encoder.ae.encode(observations).latents  # [B, 4, 12, 12]
            B = latents.shape[0]
            obs = latents.view(B, 4, -1).transpose(1, 2)  # [B, 144, 4]

        batch_size = obs.shape[0]
        self.update_history(obs)

        while len(self.obs_history) < self.condition_frames + 1:
            if len(self.obs_history) > 0:
                first_obs = self.obs_history[0]
                if first_obs.shape[0] != batch_size:
                    first_obs = first_obs.repeat(batch_size, 1, 1)
                self.obs_history.insert(0, first_obs)
            else:
                dummy_obs = torch.zeros(
                    batch_size, self.actual_img_tokens_size, 4, device=obs.device, dtype=obs.dtype
                )
                self.obs_history.append(dummy_obs)

        adjusted_history = []
        for hist_obs in self.obs_history[-self.condition_frames - 1 :]:
            if hist_obs.shape[0] != batch_size:
                if hist_obs.shape[0] < batch_size:
                    hist_obs = hist_obs.repeat(batch_size, 1, 1)
                else:
                    hist_obs = hist_obs[:batch_size]
            adjusted_history.append(hist_obs)

        feature_total = torch.stack(adjusted_history, dim=1)

        if len(self.action_history) < self.condition_frames + 1:
            dummy_actions = []
            for i in range(self.action_dim):
                min_val, max_val = self.action_ranges[i]
                dummy_action = (
                    torch.zeros(batch_size, self.condition_frames + 1, device=obs.device)
                    * (max_val - min_val)
                    / 2
                )
                dummy_actions.append(dummy_action)
            action_values_total = torch.stack(dummy_actions, dim=-1)
        else:
            adjusted_action_history = []
            for hist_action in self.action_history[-self.condition_frames - 1 :]:
                if hist_action.shape[0] != batch_size:
                    if hist_action.shape[0] < batch_size:
                        hist_action = hist_action.repeat(batch_size, 1)
                    else:
                        hist_action = hist_action[:batch_size]
                adjusted_action_history.append(hist_action)
            action_values_total = torch.stack(adjusted_action_history, dim=1)

        return feature_total, action_values_total

    def forward(self, observations):
        """Forward pass for sequence of observations

        Args:
            observations: Tensor of shape (batch_size, seq_len, img_tokens_size, vae_emb_dim)
                         or (batch_size, img_tokens_size, vae_emb_dim)

        Returns:
            Tuple: (encoded features from SpatialTemporalTransformer, None) for compatibility
        """
        feature_total, action_values_total = self._prepare_stt_input(observations)
        # feature_total: [B, condition_frames+1, 144, 4] - Image features with spatial structure
        # action_values_total: [B, condition_frames+1, action_dim] - Action history

        stt_output = self.stt(feature_total, action_values_total)
        # stt_output: [B, F, total_tokens_size, n_embd] - Full spatial-temporal embeddings

        # Take mean over frames and tokens to get global representation
        # [B, F, total_tokens_size, n_embd] -> [B, n_embd]
        global_emb = stt_output.mean(dim=(1, 2))

        # Project to match AE output dimension
        projected_output = self.output_projection(global_emb)
        return projected_output, None
