import re

import numpy as np
import torch
import torchvision.transforms as T
from mamba_ssm.utils.generation import InferenceParams
from peft import LoraConfig, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from unsloth import FastVisionModel

from .for_mmmamba.modeling_mmMamba_chat import mmMambaChatModel

ACTION_PROMPT = (
    "You control the red car in CarRacing-v3 (top-down). Stay on the gray road and avoid going onto the green grass; hug the road center when possible. "
    "Action space: steer [-1, +1] where -1 is full left and +1 is full right; accel [-1, +1] where positive is gas and negative is brake. "
    "Typical actions: Turn Left -> steer=-0.20, accel=0.00; Turn Right -> steer=0.20, accel=0.00; Go Straight -> steer=0.00, accel=0.10; Slow Down -> steer=0.00, accel=-0.10. "
    "Respond in the exact format: 'Action: steer=X.XX, accel=X.XX' using decimal values within range."
)


def load_model(
    model_id: str,
    use_quantization: bool,
    use_lora: bool,
    device: torch.device,
    use_unsloth: bool,
) -> tuple[nn.Module, AutoProcessor]:
    """Load Qwen-VL model and processor."""
    if use_unsloth:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_id,
            max_seq_length=16384,
            load_in_4bit=use_quantization,
            fast_inference=False,
            gpu_memory_utilization=0.8,
        )

        if use_lora:
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=True,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
                use_gradient_checkpointing="unsloth",
            )

        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir="./cache",
            device_map=device,
        )
        if tokenizer is not None and hasattr(tokenizer, "special_tokens_map"):
            processor.tokenizer = tokenizer

        # Enable gradient checkpointing to reduce memory usage
        model.gradient_checkpointing_enable()
        return model, processor

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        cache_dir="./cache",
        device_map=device,
    )
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="./cache",
        device_map=device,
    )

    return model, processor


def prepare_vlm_inputs(
    processor: AutoProcessor,
    images: torch.Tensor,
    rewards: torch.Tensor,
    task_prompt: str,
) -> dict[str, torch.Tensor]:
    """Build VLM messages and prepare model inputs.

    Args:
        processor: AutoProcessor instance
        images: (B, T, C, H, W) tensor
        rewards: (B, T, 1) tensor
        task_prompt: Task prompt string (can be empty)

    Returns:
        Dictionary of model inputs
    """
    device = images.device
    batch_size, seq_len = images.shape[:2]
    messages = []
    for b in range(batch_size):
        content: list[dict] = []
        if task_prompt:
            content.append({"type": "text", "text": task_prompt})
        for t in range(seq_len):
            img_tensor = images[b, t].to(torch.float32)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            content.append({"type": "image", "image": Image.fromarray(img_np)})
            reward_text = f"reward {float(rewards[b, t, 0]):.2f}"
            content.append({"type": "text", "text": reward_text})
        messages.append([{"role": "user", "content": content}])

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    proc_images, videos, video_kwargs = process_vision_info(
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

    inputs = processor(
        text=text,
        images=proc_images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        padding=True,
        **video_kwargs,
    )
    inputs.pop("token_type_ids", None)
    inputs = {
        k: v.to(device).to(torch.bfloat16) if v.dtype.is_floating_point else v.to(device)
        for k, v in inputs.items()
    }
    return inputs


def parse_action_text(action_text: str) -> np.ndarray:
    """Parse action text and extract steer, accel values.

    Args:
        action_text: Text in format 'Action: steer=X.XX, accel=X.XX'

    Returns:
        np.ndarray of shape (2,) containing [steer, accel]

    Example:
        >>> parse_action_text("Action: steer=0.5, accel=0.3")
        array([0.5, 0.3])
    """
    steer, accel = 0.0, 0.0

    steer_match = re.search(r"steer=([+-]?\d*\.?\d+)", action_text)
    accel_match = re.search(r"accel=([+-]?\d*\.?\d+)", action_text)

    steer = float(steer_match.group(1)) if steer_match else 0.0
    accel = float(accel_match.group(1)) if accel_match else 0.0

    action_array = np.array([steer, accel], dtype=np.float32)
    action_array[0] = np.clip(action_array[0], -1.0, 1.0)
    action_array[1] = np.clip(action_array[1], -1.0, 1.0)

    return action_array


class QwenVLEncoder(nn.Module):
    def __init__(
        self,
        output_text: bool,
        use_quantization: bool,
        use_lora: bool,
        target_layer_idx: int,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.output_text = output_text
        self.use_lora = use_lora
        self.target_layer_idx = target_layer_idx
        self.seq_len = seq_len

        device = torch.device("cuda")

        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        self.model, self.processor = load_model(
            model_id,
            use_quantization,
            use_lora,
            device,
            use_unsloth=False,
        )

        out_dim = 4
        self.out_proj = nn.Linear(2048, out_dim)
        self.device = device
        self.out_proj = self.out_proj.to(device)
        self.video_fps = 50 / 8
        self._dummy_state = torch.zeros(1, 1, 1)

        # Compute output_dim dynamically with dummy forward pass
        self.output_dim = self._compute_output_dim()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _compute_output_dim(self) -> int:
        """Compute output dimension by running a dummy forward pass."""
        dummy_images = torch.zeros(1, self.seq_len, 3, 96, 96, device=self.device)
        dummy_rewards = torch.zeros(1, self.seq_len, 1, device=self.device)

        model_inputs = prepare_vlm_inputs(self.processor, dummy_images, dummy_rewards, "")

        output = self.model.forward(**model_inputs, output_hidden_states=True)
        hidden = output["hidden_states"][self.target_layer_idx]

        x = hidden.to(torch.float32)
        x = self.out_proj(x)
        x = x.flatten(start_dim=1)
        return x.shape[1]

    def init_state(self) -> torch.Tensor:
        return self._dummy_state.clone()

    def forward(
        self,
        images: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        with torch.enable_grad() if self.use_lora else torch.no_grad():
            model_inputs = prepare_vlm_inputs(self.processor, images, rewards, "")

            output = self.model.forward(**model_inputs, output_hidden_states=True)
            hidden = output["hidden_states"][self.target_layer_idx]

        x = hidden.to(torch.float32)
        x = self.out_proj(x)
        x = x.flatten(start_dim=1)

        return x, rnn_state


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
            dtype=torch.bfloat16,
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

    def init_state(self) -> torch.Tensor:
        """Return dummy state for compatibility with standard encoders"""
        return torch.zeros(1, 1, 1)

    def forward(
        self,
        images: torch.Tensor,
        obs_z: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        rnn_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        # images: (B, T, C, H, W)
        # obs_z, actions, rewards are ignored for VLM encoders
        # rnn_state is passed through unchanged (VLM encoders are stateless)

        batch_size = images.shape[0]
        device = images.device

        # MMMamba processes the last frame from each batch sample
        # Stack all last frames into a batch: (B, C, H, W)
        last_frames = images[:, -1, :, :, :]  # (B, C, H, W)

        # Transform all frames at once
        batch_images = torch.stack([self.transform(last_frames[b]) for b in range(batch_size)])
        batch_images = batch_images.to(device).to(torch.bfloat16)  # (B, C, H, W)

        # Create tokenized input (same for all batch samples)
        prompt = (
            ACTION_PROMPT + " <|im_start|>" + "<IMG_CONTEXT>" * self.image_token_num + "<|im_end|>"
        )
        messages = [{"role": "user", "content": prompt}]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Repeat input_ids for batch
        input_ids = model_inputs["input_ids"].repeat(batch_size, 1).to(device)

        stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        stop_token_ids = [tid for tid in stop_token_ids if tid is not None]

        # Note: inference_params with batch > 1 may not be supported
        # Process first batch only for now due to inference_params limitation
        self.inference_params.seqlen_offset = 0

        outputs = self.model.forward(
            input_ids=input_ids[0:1],
            pixel_values=batch_images[0:1],
            inference_params=self.inference_params,
            output_hidden_states=True,
        )

        # Get representation from first batch
        x_first = outputs["hidden_states"][-1][:, 0].to(torch.float32)

        # For other batches, process without inference_params
        batch_outputs = [x_first]
        for b in range(1, batch_size):
            outputs_b = self.model.forward(
                input_ids=input_ids[b : b + 1],
                pixel_values=batch_images[b : b + 1],
                inference_params=None,
                output_hidden_states=True,
            )
            x_b = outputs_b["hidden_states"][-1][:, 0].to(torch.float32)
            batch_outputs.append(x_b)

        batch_output = torch.cat(batch_outputs, dim=0)  # (B, output_dim)

        # Generate action text for first batch only
        output_ids = []
        logits = outputs["logits"]
        last_logit = logits[:, -1, :]
        token = torch.argmax(last_logit, dim=-1)

        self.inference_params.seqlen_offset += input_ids.shape[1]
        input_ids_gen = token.unsqueeze(1)

        for _ in range(50):
            if token.item() in stop_token_ids:
                break
            output_ids.append(token.item())

            outputs = self.model.forward(
                input_ids=input_ids_gen,
                inference_params=self.inference_params,
            )
            logits = outputs["logits"]
            last_logit = logits[:, -1, :]
            token = torch.argmax(last_logit, dim=-1)
            input_ids_gen = token.unsqueeze(1)

        action_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return batch_output, rnn_state, action_text
