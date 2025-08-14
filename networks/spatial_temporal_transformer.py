import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def compute_axial_cis(dim, end_x, end_y, theta=1000.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t_x = torch.arange(end_x, device=freqs_x.device)
    t_y = torch.arange(end_y, device=freqs_y.device)

    freqs_x = torch.outer(t_x, freqs_x).float()
    freqs_y = torch.outer(t_y, freqs_y).float()

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    freqs_cis = []
    for i in range(end_x):
        for j in range(end_y):
            freqs_cis.append(torch.cat([freqs_cis_x[i], freqs_cis_y[j]], dim=-1))

    return torch.stack(freqs_cis)


def get_fourier_embeds_from_coordinates(embed_dim, coords):
    device = coords.device
    dtype = coords.dtype

    if coords.dim() == 2:
        coords = coords.unsqueeze(-1)

    batch_size, seq_len, coord_dim = coords.shape

    half_dim = embed_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)

    emb = emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    coords = coords.unsqueeze(-1)

    emb = coords * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = emb.view(batch_size, seq_len, coord_dim, embed_dim)

    return emb


class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.action_tokens_num = config.token_size_dict["action_tokens_size"]
        self.img_tokens_num = config.token_size_dict["img_tokens_size"]
        self.total_tokens_num = config.token_size_dict["total_tokens_size"]
        self.patch_size = config.patch_size
        self.num_tokens = self.total_tokens_num
        self.freqs_cis_singlescale = compute_axial_cis(
            dim=config.n_embd // self.n_head,
            end_x=self.patch_size[0],
            end_y=self.patch_size[1],
            theta=1000.0,
        )

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.to(q.dtype)
        y = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout_rate
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.resid_drop(self.proj(y))
        return y


class CausalSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x


class SpaceSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.action_tokens_num = config.token_size_dict["action_tokens_size"]
        self.img_tokens_num = config.token_size_dict["img_tokens_size"]
        self.total_tokens_num = config.token_size_dict["total_tokens_size"]
        self.patch_size = config.patch_size
        self.num_tokens = self.total_tokens_num
        self.freqs_cis_singlescale = compute_axial_cis(
            dim=config.n_embd // self.n_head,
            end_x=self.patch_size[0],
            end_y=self.patch_size[1],
            theta=1000.0,
        )

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = (
            F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_rate)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.resid_drop(self.proj(y))
        return y


class SpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SpaceSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        attn = self.attn(self.ln1(x))
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x


class CausalTimeSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.attn_dropout_rate = config.attn_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.qk_norm = True

        if self.qk_norm:
            self.q_norm = nn.LayerNorm(config.n_embd)
            self.k_norm = nn.LayerNorm(config.n_embd)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, attn_mask):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = (
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask.to(q.dtype), dropout_p=self.attn_dropout_rate
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.resid_drop(self.proj(y))
        return y


class CausalTimeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalTimeSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask):
        attn = self.attn(self.ln1(x), attn_mask)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTimeSpaceBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_time_block = CausalTimeBlock(config)
        self.space_block = SpaceBlock(config)

    def forward(self, x, attn_mask):
        b, f, l, c = x.shape
        x = rearrange(x, "b f l c -> (b l) f c")
        x = self.causal_time_block(x, attn_mask)
        x = rearrange(x, "(b l) f c -> (b f) l c", b=b, l=l, f=f)
        x = self.space_block(x)
        x = rearrange(x, "(b f) l c -> b f l c", b=b, f=f)
        return x


class SpatialTemporalTransformer(nn.Module):
    def __init__(
        self,
        block_size,
        n_layer,
        n_head,
        n_embd,
        embd_pdrop,
        resid_pdrop,
        attn_pdrop,
        n_unmasked,
        local_rank,
        condition_frames,
        latent_size,
        token_size_dict,
        vae_emb_dim,
        temporal_block,
        action_ranges,
    ):
        super().__init__()
        config = GPTConfig(
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
            patch_size=latent_size,
            condition_frames=condition_frames,
            token_size_dict=token_size_dict,
        )

        self.C = n_embd
        self.Cvae = vae_emb_dim
        self.action_emb_dim = 512
        self.action_ranges = action_ranges
        self.num_actions = len(action_ranges)
        self.latent_size = latent_size
        self.temporal_block = temporal_block
        self.img_projector = nn.Sequential(
            nn.Linear(self.Cvae, self.C // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.C // 2, self.C, bias=False),
            nn.LayerNorm(self.C),
        )
        self.action_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.action_emb_dim, self.action_emb_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(self.action_emb_dim, self.C, bias=False),
                    nn.LayerNorm(self.C),
                )
                for _ in range(self.num_actions)
            ]
        )
        self.causal_time_space_num = config.n_layer[0]
        self.auto_regressive_num = config.n_layer[1]
        print(
            "self.causal_time_space_num, self.auto_regressive_num",
            self.causal_time_space_num,
            self.auto_regressive_num,
        )

        self.local_rank = local_rank
        self.img_token_size = token_size_dict["img_tokens_size"]
        self.total_token_size = token_size_dict["total_tokens_size"]
        self.action_token_size = token_size_dict["action_tokens_size"]
        self.prefix_size = self.total_token_size - self.img_token_size
        self.condition_frames = condition_frames

        self.time_emb = nn.Parameter(torch.zeros(50, self.C))
        nn.init.normal_(self.time_emb.data, mean=0, std=0.02)
        self.begin_ends = []

        self.causal_time_space_blocks = nn.Sequential(
            *[CausalTimeSpaceBlock(config) for _ in range(self.causal_time_space_num)]
        )

        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

        matrix = torch.tril(torch.ones(condition_frames, condition_frames))
        time_causal_mask = torch.where(matrix == 0, float("-inf"), matrix)
        time_causal_mask = torch.where(matrix == 1, 0, time_causal_mask)
        self.register_buffer("mask_time", time_causal_mask.contiguous())

        matrix_1 = torch.ones(self.total_token_size, self.total_token_size)
        for i in range(0, self.prefix_size):
            matrix_1[i, self.prefix_size :] = 0
        seq_causal_mask = torch.where(matrix_1 == 0, float("-inf"), matrix_1)
        seq_causal_mask = torch.where(matrix_1 == 1, 0, seq_causal_mask)
        beta = 0.1
        space_weight = torch.zeros(self.total_token_size, self.total_token_size)
        space_weight[:, 0] = 2
        space_weight[:, 1] = 1
        seq_causal_mask = seq_causal_mask + space_weight * beta
        self.register_buffer("mask_space", seq_causal_mask.contiguous())

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_action_emb(self, action_values):
        """
        Convert action float values to embeddings.

        Args:
            action_values: [B, T, num_actions] - Float values for each action
        """
        action_values_normalize = []
        for i in range(self.num_actions):
            # Normalize float values to [0, 1] range using min/max of each action
            min_val, max_val = self.action_ranges[i]
            normalized = (action_values[:, :, i : i + 1] - min_val) / (max_val - min_val)
            # Clamp to [0, 1] to ensure values are in valid range
            normalized = torch.clamp(normalized, 0.0, 1.0)
            action_values_normalize.append(normalized)

        combined_values = torch.cat(action_values_normalize, dim=-1)
        action_emb = get_fourier_embeds_from_coordinates(self.action_emb_dim, combined_values)

        action_embs = torch.split(action_emb, dim=2, split_size_or_sections=1)
        return action_embs

    def forward(self, feature_total, action_values_total):
        """
        Forward pass of the SpatialTemporalTransformer.

        Args:
            feature_total: [B, F+1, img_tokens_size, vae_emb_dim] - Image features
            action_values_total: [B, (F+1)*temporal_block, num_actions] - Action float values

        Returns:
            dict: Contains 'logits' and 'action_emb'
        """
        B, F, _, _ = feature_total.shape
        F = F - 1
        action_embs = self.get_action_emb(action_values_total)
        action_token_embeddings = []
        for i, emb in enumerate(action_embs):
            action_token_embeddings.append(self.action_projectors[i](emb))
        feature_embeddings = self.img_projector(feature_total)

        action_token_embeddings_reshaped = []
        for action_emb in action_token_embeddings:
            reshaped = rearrange(
                action_emb, "B (F T) L C -> B F (L T) C", F=F + 1, T=self.temporal_block
            )
            action_token_embeddings_reshaped.append(reshaped)

        input_action_token_embeddings = []
        for action_emb in action_token_embeddings_reshaped:
            input_action_token_embeddings.append(action_emb[:, :-1, ...])
        input_feature_embeddings = feature_embeddings[:, :-1, ...]

        combined_token_embeddings = torch.cat(
            input_action_token_embeddings + [input_feature_embeddings],
            dim=2,
        )

        time_emb_F = self.time_emb[:F, :].unsqueeze(0)
        time_emb_F = torch.repeat_interleave(
            time_emb_F[:, :, None, :], self.total_token_size, dim=2
        )

        time_space_token_embeddings = combined_token_embeddings + time_emb_F

        for i in range(self.causal_time_space_num):
            time_space_token_embeddings = self.causal_time_space_blocks[i](
                time_space_token_embeddings, self.mask_time
            )
        auto_regressive_token_embeddings = rearrange(
            time_space_token_embeddings, "B F L C -> (B F) L C", B=B, F=F
        )

        target_action_embeddings = []
        for action_emb in action_token_embeddings_reshaped:
            target_emb = action_emb[:, self.temporal_block :, ...].reshape(
                B * F, self.temporal_block * self.C
            )
            target_action_embeddings.append(target_emb)

        action_emb = torch.cat(target_action_embeddings, dim=1)

        out = {"logits": auto_regressive_token_embeddings, "action_emb": action_emb}
        return out

    @torch.no_grad()
    def evaluate(self, feature, action_values_total, sample_last=True):
        """
        Evaluate the model.

        Args:
            feature: [B, F, img_tokens_size, vae_emb_dim] - Image features
            action_values_total: [B, F*temporal_block, num_actions] - Action float values
            sample_last: bool - Whether to sample only the last frame

        Returns:
            tuple: (logits, action_emb)
        """
        action_embs = self.get_action_emb(action_values_total)
        action_token_embeddings = []
        for i, emb in enumerate(action_embs):
            action_token_embeddings.append(self.action_projectors[i](emb))

        input_action_token_embeddings = []
        for action_emb in action_token_embeddings:
            input_action_token_embeddings.append(action_emb[:, :-1, ...])

        feature_embeddings = self.img_projector(feature)
        combined_token_embeddings = torch.cat(
            input_action_token_embeddings + [feature_embeddings],
            dim=2,
        )
        B, F, _, _ = combined_token_embeddings.shape

        time_emb_F = self.time_emb[:F, :].unsqueeze(0)
        time_emb_F = torch.repeat_interleave(
            time_emb_F[:, :, None, :], self.total_token_size, dim=2
        )

        time_space_token_embeddings = combined_token_embeddings + time_emb_F

        for i in range(self.causal_time_space_num):
            time_space_token_embeddings = self.causal_time_space_blocks[i](
                time_space_token_embeddings, self.mask_time
            )

        if sample_last:
            time_space_token_embeddings = time_space_token_embeddings[:, -1:, :, :]
            target_action_embeddings = []
            for action_emb in action_token_embeddings:
                target_emb = action_emb[:, -1, ...].reshape(B, self.C)
                target_action_embeddings.append(target_emb)
        else:
            target_action_embeddings = []
            for action_emb in action_token_embeddings:
                target_emb = action_emb[:, 1:, ...].reshape(B * F, self.C)
                target_action_embeddings.append(target_emb)

        auto_regressive_token_embeddings = rearrange(
            time_space_token_embeddings, "B F L C -> (B F) L C"
        )
        action_emb = torch.cat(target_action_embeddings, dim=1)
        return auto_regressive_token_embeddings, action_emb

    def get_action_emb_single(self, action_values):
        action_embs = self.get_action_emb(action_values)
        action_token_embeddings = []
        for i, emb in enumerate(action_embs):
            action_token_embeddings.append(self.action_projectors[i](emb))
        action_emb = torch.cat(action_token_embeddings, dim=-1).reshape(
            -1, self.C * self.num_actions
        )
        return action_emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
