import torch
import torch.nn as nn

from .sparse_utils import apply_one_shot_pruning


class SequenceProcessor(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, sparsity: float):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Positional Encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len * 3 - 1, self.hidden_dim), requires_grad=True
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            bias=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.sparse_mask = (
            None
            if sparsity == 0.0
            else apply_one_shot_pruning(self.transformer_encoder, overall_sparsity=sparsity)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch_size, seq_len, hidden_dim)

        Returns:
            x (torch.Tensor): (batch_size, seq_len, hidden_dim)
        """
        # 処理
        x = x + self.pos_embedding

        # Causal Maskの生成
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=None)

        return x
