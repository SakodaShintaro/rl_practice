# Copyright (c) The mmMamba team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

compute_ARank = False  # [ARank] Set this to True to compute attention rank

from .configuration_mmMamba_embedding import mmMambaEmbeddingConfig
from .modeling_mmMamba import (
    MHA_LM,
    Mamba2_LM,
    mmMambaDecoderLayer,
    mmMambaRotaryEmbedding,
)

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

import torch.nn.functional as F

logger = logging.get_logger(__name__)

flash_attn_func, flash_attn_varlen_func = None, None
pad_input, index_first_axis, unpad_input = None, None, None


def _import_flash_attn():
    global flash_attn_func, flash_attn_varlen_func
    global pad_input, index_first_axis, unpad_input
    try:
        from flash_attn import (
            flash_attn_func as _flash_attn_func,
        )
        from flash_attn import (
            flash_attn_varlen_func as _flash_attn_varlen_func,
        )
        from flash_attn.bert_padding import (
            index_first_axis as _index_first_axis,
        )
        from flash_attn.bert_padding import (
            pad_input as _pad_input,
        )
        from flash_attn.bert_padding import (
            unpad_input as _unpad_input,
        )

        flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
        pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    except ImportError:
        raise ImportError("flash_attn is not installed.")


_import_flash_attn()


mmMamba_ATTENTION_CLASSES = {"mha": MHA_LM, "mamba2": Mamba2_LM}


class VisionEmbeddings(nn.Module):
    def __init__(self, config: mmMambaEmbeddingConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=self.config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

        self.post_init()

    def post_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        use_cls_token=False,
    ) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = pixel_values.to(target_dtype)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if use_cls_token:
            class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
            embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
            assert not self.config.use_2d_sincos_pos_embed, (
                "2D SinCos pos embed is not supported with use_cls_token"
            )
            position_embedding = torch.cat(
                [
                    self.position_embedding[:, :1, :],
                    self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
                ],
                dim=1,
            )
            embeddings = embeddings + position_embedding
        else:
            position_embedding = self._get_pos_embed(
                self.position_embedding[:, 1:, :], height, width
            ).to(target_dtype)
            embeddings = patch_embeds + position_embedding

        return embeddings


class mmMambaEmbedding(PreTrainedModel):
    config_class = mmMambaEmbeddingConfig
    _supports_flash_attn_2 = True

    def __init__(self, config: mmMambaEmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.gradient_checkpointing = True

        self.vision_embeddings = VisionEmbeddings(config)
        self.llm_text_embeddings = nn.Embedding(
            self.config.llm_vocab_size, self.config.llm_hidden_size
        )
        self.special_token_maps = config.special_token_maps
        if len(self.special_token_maps) > 0:
            self.special_text_embeddings = nn.Embedding(
                len(config.special_token_maps), self.config.llm_hidden_size
            )

        assert self.config.use_ls is False, "LS is not supported in mmMamba"
        self.encoder = nn.ModuleList(
            [mmMambaDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )

        if self.config.use_pixel_shuffle_proj:
            self.pixel_shuffle_proj = nn.Sequential(
                nn.Linear(
                    int(config.hidden_size / (config.downsample_ratio * config.downsample_ratio)),
                    config.hidden_size,
                ),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

        self.num_img_tokens = (self.config.image_size // self.config.patch_size) ** 2

    def set_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for layer in self.encoder:
            layer.gradient_checkpointing = True

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.vision_embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = (
            pos_emb[:, 1:, :]
            .reshape(1, old_size // patch_size, old_size // patch_size, -1)
            .permute(0, 3, 1, 2)
        )
        pos_emb = F.interpolate(
            pos_emb.float(), size=new_size // patch_size, mode="bicubic", align_corners=False
        )
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.vision_embeddings.position_embedding = nn.Parameter(pos_emb)
        self.vision_embeddings.image_size = new_size
        logger.info("Resized position embeddings from {} to {}".format(old_size, new_size))

    def replace_img_tokens(self, input_ids, hidden_states, vision_hidden_states):
        img_context_token_mask = input_ids == self.config.img_context_token_id
        hidden_states[img_context_token_mask] = hidden_states[
            img_context_token_mask
        ] * 0.0 + vision_hidden_states.flatten(0, 1)

        return hidden_states

    def get_text_mask(self, input_ids):
        txt_mask = input_ids != self.config.img_context_token_id

        return txt_mask

    def get_input_embeddings(self, input_ids):
        special_mask = input_ids > self.llm_text_embeddings.weight.shape[0] - 1
        llm_embeddings = self.llm_text_embeddings(input_ids * (~special_mask).to(input_ids))

        if len(self.special_token_maps) > 0:
            special_embeddings = self.special_text_embeddings(
                (input_ids - self.llm_text_embeddings.weight.shape[0]) * special_mask.to(input_ids)
            )
            special_mask = special_mask.unsqueeze(-1)
            text_embeddings = llm_embeddings * (~special_mask).to(
                llm_embeddings
            ) + special_embeddings * special_mask.to(llm_embeddings)
        else:
            text_embeddings = llm_embeddings

        return text_embeddings

    def get_txt_embeddings(self, input_ids):
        B, L = input_ids.shape
        txt_mask = input_ids != self.config.img_context_token_id
        txt_embeddings = self.llm_text_embeddings(input_ids[txt_mask])
        txt_embeddings = txt_embeddings.reshape(-1, txt_embeddings.shape[-1])

        return txt_embeddings

    def get_txt_feature(self, input_ids, feature):
        B, L, C = feature.shape
        txt_mask = input_ids != self.config.img_context_token_id
        txt_feature = feature[txt_mask].reshape(-1, C)

        return txt_feature

    def get_img_feature(self, input_ids, feature):
        B, L, C = feature.shape
        img_mask = input_ids == self.config.img_context_token_id
        img_feature = feature[img_mask].reshape(-1, C)

        return img_feature

    def pixel_shuffle(self, x, scale_factor=0.5):
        if getattr(self.config, "pixel_shuffle_loc", "pre") == "post":
            x = x.view(x.shape[0] // self.num_img_tokens, self.num_img_tokens, -1)

        n, l, c = x.size()
        h = w = int(l**0.5)
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor))
        )
        x = (
            x.permute(0, 2, 1, 3)
            .reshape(
                n, int(l * scale_factor * scale_factor), int(c / (scale_factor * scale_factor))
            )
            .contiguous()
        )

        if getattr(self.config, "pixel_shuffle_loc", "pre") == "post":
            x = x.view(
                int(x.shape[0] * self.num_img_tokens * (self.config.downsample_ratio**2)), -1
            )
        return x

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inference_params=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = True,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is not None:
            if len(pixel_values.shape) == 4:
                if self.gradient_checkpointing and self.training:
                    vision_hidden_states = torch.utils.checkpoint.checkpoint(
                        self.vision_embeddings, pixel_values
                    )
                else:
                    vision_hidden_states = self.vision_embeddings(pixel_values)

                if (
                    self.config.use_pixel_shuffle_proj
                    and getattr(self.config, "pixel_shuffle_loc", "pre") == "pre"
                ):
                    vision_hidden_states = self.pixel_shuffle(
                        vision_hidden_states, scale_factor=self.config.downsample_ratio
                    )
                    if self.gradient_checkpointing and self.training:
                        vision_hidden_states = torch.utils.checkpoint.checkpoint(
                            self.pixel_shuffle_proj, vision_hidden_states
                        )
                    else:
                        vision_hidden_states = self.pixel_shuffle_proj(vision_hidden_states)

                hidden_states = self.get_input_embeddings(input_ids)
                hidden_states = self.replace_img_tokens(
                    input_ids, hidden_states, vision_hidden_states
                )
            else:
                raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        for layer_idx, layer_module in enumerate(self.encoder):
            if self.gradient_checkpointing and self.training:
                assert use_cache is None, "Gradient checkpointing is not compatible with cache"
                outputs = torch.utils.checkpoint.checkpoint(
                    layer_module,
                    hidden_states,
                    inference_params,
                    None,
                    False,
                    False,
                )
                hidden_states = outputs[0]
            else:
                outputs = layer_module(
                    hidden_states=hidden_states,
                    inference_params=inference_params,
                    use_cache=use_cache,
                )
                hidden_states = outputs[0]

        img_feature = self.get_img_feature(input_ids, hidden_states)

        if (
            self.config.use_pixel_shuffle_proj
            and getattr(self.config, "pixel_shuffle_loc", "pre") == "post"
        ):
            img_feature = self.pixel_shuffle(img_feature, scale_factor=self.config.downsample_ratio)
            img_feature = self.pixel_shuffle_proj(img_feature)

        return img_feature, hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            layer.layer_idx: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for layer in self.encoder
        }
