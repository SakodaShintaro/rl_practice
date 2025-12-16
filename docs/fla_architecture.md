# FLAライブラリのアーキテクチャと抽象化

このドキュメントでは、FLA (Flash Linear Attention) ライブラリにおけるブロックやネットワークの抽象化・共通化、およびCache統一インターフェースについて説明します。

## 目次

1. [レイヤーレベルの統一インターフェース](#1-レイヤーレベルの統一インターフェース)
2. [ブロックレベルでの切り替え機構](#2-ブロックレベルでの切り替え機構)
3. [Configurationによる制御](#3-configurationによる制御)
4. [実装の共通パターン](#4-実装の共通パターン)
5. [複数のブロックを切り替える実装方法](#5-複数のブロックを切り替える実装方法)
6. [学習時・推論時の継続的なCache使用](#6-学習時推論時の継続的なcache使用)
7. [実装例](#7-実装例)

---

## 1. レイヤーレベルの統一インターフェース

FLAでは、すべての注意機構レイヤー(Attention、GatedDeltaNet、Mamba2など)が**共通のforwardシグネチャ**を持っています。

### 統一されたforwardシグネチャ

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]
```

### 特徴

- **入力**: `hidden_states` (バッチサイズ×シーケンス長×隠れ層サイズ)
- **出力**: タプル `(output, attentions, past_key_values)`
- **共通パラメータ**: `attention_mask`, `past_key_values`, `use_cache`など

この統一により、異なるレイヤータイプを同じインターフェースで扱うことができます。

**参考ファイル:**
- `.venv/lib/python3.10/site-packages/fla/layers/gated_deltanet.py:203`
- `.venv/lib/python3.10/site-packages/fla/layers/attn.py:80`
- `.venv/lib/python3.10/site-packages/fla/layers/mamba2.py:599`

---

## 2. ブロックレベルでの切り替え機構

`GatedDeltaNetBlock` では、ブロック内部で**条件分岐による切り替え**が実装されています。

### 実装例

```python
# fla/models/gated_deltanet/modeling_gated_deltanet.py:46-71
if config.attn is not None and layer_idx in config.attn['layers']:
    # 特定のレイヤーでAttentionを使用
    self.attn = Attention(
        hidden_size=config.hidden_size,
        num_heads=config.attn['num_heads'],
        num_kv_heads=config.attn['num_kv_heads'],
        qkv_bias=config.attn['qkv_bias'],
        window_size=config.attn['window_size'],
        rope_theta=config.attn['rope_theta'],
        max_position_embeddings=config.max_position_embeddings,
        layer_idx=layer_idx
    )
else:
    # それ以外ではGatedDeltaNetを使用
    self.attn = GatedDeltaNet(
        mode=config.attn_mode,
        hidden_size=config.hidden_size,
        expand_v=config.expand_v,
        head_dim=config.head_dim,
        num_heads=config.num_heads,
        num_v_heads=config.num_v_heads,
        use_gate=config.use_gate,
        use_short_conv=config.use_short_conv,
        allow_neg_eigval=config.allow_neg_eigval,
        conv_size=config.conv_size,
        norm_eps=config.norm_eps,
        layer_idx=layer_idx
    )
```

このパターンにより、同じ`self.attn`という属性名で異なるレイヤーを格納できます。

---

## 3. Configurationによる制御

`GatedDeltaNetConfig` では、`attn` パラメータで特定レイヤーにAttentionを挿入可能です。

### 設定例

```python
# fla/models/gated_deltanet/configuration_gated_deltanet.py
config = GatedDeltaNetConfig(
    hidden_size=2048,
    num_heads=6,
    attn={
        'layers': [0, 5, 10],  # レイヤー0, 5, 10でAttentionを使用
        'num_heads': 32,
        'num_kv_heads': 32,
        'window_size': 2048,
        'rope_theta': 10000.0
    }
)
```

---

## 4. 実装の共通パターン

各レイヤーは以下の構造を持っています:

### 共通コンポーネント

- **Projection layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Normalization**: `RMSNorm` または `FusedRMSNormGated`
- **Optional components**:
  - Convolution (`ShortConvolution`)
  - Gate (`g_proj`)
  - Position encoding (Attentionのみ: `RotaryEmbedding`)

### レイヤーごとの特徴

| レイヤー | 内部状態 | Conv | Gate | Positional Encoding |
|---------|---------|------|------|---------------------|
| Attention | KVキャッシュ | × | × | RoPE |
| GatedDeltaNet | recurrent_state | ○ | ○ | × |
| Mamba2 | SSM state | ○ | ○ | × |

---

## 5. 複数のブロックを切り替える実装方法

### 方法1: Configベースの動的切り替え（推奨）

```python
class UnifiedConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=2048,
        num_heads=6,
        layer_configs=None,  # レイヤーごとの設定
        **kwargs
    ):
        # layer_configs = [
        #   {'type': 'attention', 'num_heads': 32},
        #   {'type': 'gated_deltanet', 'head_dim': 256},
        #   {'type': 'mamba2', 'state_size': 128},
        # ]
        self.layer_configs = layer_configs or []
        super().__init__(**kwargs)

class UnifiedBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        layer_config = config.layer_configs[layer_idx]

        # レイヤータイプに応じて動的に選択
        if layer_config['type'] == 'attention':
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=layer_config.get('num_heads', config.num_heads),
                layer_idx=layer_idx
            )
        elif layer_config['type'] == 'gated_deltanet':
            self.attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                head_dim=layer_config.get('head_dim', 256),
                layer_idx=layer_idx
            )
        elif layer_config['type'] == 'mamba2':
            self.attn = Mamba2(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                state_size=layer_config.get('state_size', 128),
                layer_idx=layer_idx
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_config['type']}")
```

### 方法2: Factoryパターン

```python
class LayerFactory:
    @staticmethod
    def create_layer(layer_type, config, layer_idx):
        layer_map = {
            'attention': Attention,
            'gated_deltanet': GatedDeltaNet,
            'mamba2': Mamba2,
        }

        if layer_type not in layer_map:
            raise ValueError(f"Unknown layer type: {layer_type}")

        return layer_map[layer_type](
            hidden_size=config.hidden_size,
            layer_idx=layer_idx,
            # 各レイヤー固有のパラメータを追加
        )

class UnifiedBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.attn = LayerFactory.create_layer(layer_type, config, layer_idx)
```

### 方法3: レイヤーリストによる柔軟な構成

```python
class FlexibleConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=2048,
        layer_pattern=None,  # 例: ['A', 'G', 'M', 'A', 'G', 'M', ...]
        **kwargs
    ):
        # A=Attention, G=GatedDeltaNet, M=Mamba2
        self.layer_pattern = layer_pattern or ['G'] * 12

class FlexibleBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        layer_type = config.layer_pattern[layer_idx]

        if layer_type == 'A':
            self.attn = Attention(...)
        elif layer_type == 'G':
            self.attn = GatedDeltaNet(...)
        elif layer_type == 'M':
            self.attn = Mamba2(...)
```

---

## 6. 学習時・推論時の継続的なCache使用

### FLAのCache統一インターフェース

FLAでは `FLACache` という統一的なCacheシステムがあり、以下の4種類の状態を管理できます:

```python
# fla/models/utils.py:54-59
{
    'recurrent_state': torch.Tensor,     # RNN系レイヤーの内部状態
    'attn_state': (torch.Tensor, ...),   # AttentionのKVキャッシュ
    'conv_state': (torch.Tensor, ...),   # 畳み込みの状態
    'ffn_state': Any                     # FFNの状態（将来の拡張用）
}
```

### レイヤーごとの使用状況

| レイヤー | recurrent_state | attn_state | conv_state |
|---------|----------------|------------|------------|
| GatedDeltaNet | ○ | × | ○ |
| DeltaNet | ○ | × | ○ |
| GLA | ○ | × | ○ |
| Attention | × | ○ (KVキャッシュ) | × |
| Mamba2 | × | × | 独自Cache |

### 学習時のCache利用

現在の実装では、デフォルトで学習時に `use_cache=False` になりますが、これは慣習であって技術的制約ではありません。

```python
# modeling_gated_deltanet.py:219
use_cache = use_cache if use_cache is not None else (
    self.config.use_cache if not self.training else False
)
```

重要な発見:
```python
# gated_deltanet.py:222-227
if self.training:
    assert mode == 'chunk', "Only chunk mode is supported in training."

last_state = None
if past_key_values is not None and len(past_key_values) > self.layer_idx:
    last_state = past_key_values[self.layer_idx]  # 学習時でも使用可能！
```

**つまり、明示的に `use_cache=True` を渡せば学習時でもCacheは機能します。**

### Attentionの取り扱い

Attentionは内部状態（recurrent_state）がないため、`attn_state`（KVキャッシュ）のみを使います:

- 自動的にKVキャッシュが蓄積される
- `window_size` が設定されている場合、古いキャッシュは自動的に削除される
- インターフェースは他のレイヤーと同じ

**Attentionでも統一インターフェースは保たれます:**
- 入力: `past_key_values` (FLACache)
- 出力: 更新された `past_key_values`
- 内部的には `attn_state` のみを使用

---

## 7. 実装例

### 統一的なCacheインターフェースを持つカスタムブロック

```python
from fla.layers.attn import Attention
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.mamba2 import Mamba2
from fla.models.utils import Cache
import torch.nn as nn

class UnifiedMixerBlock(nn.Module):
    """学習時・推論時ともにCacheを使える統一ブロック"""

    def __init__(
        self,
        config,
        layer_idx,
        mixer_type='gated_deltanet',  # 'attention', 'gated_deltanet', 'mamba2'
        force_cache_in_training=True
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.force_cache_in_training = force_cache_in_training

        # Normalization
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        # Mixer layer selection
        if mixer_type == 'attention':
            self.mixer = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                layer_idx=layer_idx
            )
        elif mixer_type == 'gated_deltanet':
            self.mixer = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx
            )
        elif mixer_type == 'mamba2':
            self.mixer = Mamba2(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                layer_idx=layer_idx
            )
        else:
            raise ValueError(f"Unknown mixer type: {mixer_type}")

        self.mixer_type = mixer_type
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        # MLP省略...

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        **kwargs
    ):
        # 学習時でもCacheを強制的に有効化
        if use_cache is None:
            if self.force_cache_in_training:
                use_cache = True  # 常にTrue
            else:
                use_cache = self.config.use_cache if not self.training else False

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # 統一インターフェースでmixerを呼び出し
        hidden_states, attentions, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

        hidden_states = residual + hidden_states
        # MLP処理...

        return hidden_states, attentions, past_key_values
```

### レイヤー構成をConfigで柔軟に制御

```python
class FlexibleConfig:
    def __init__(
        self,
        hidden_size=2048,
        num_heads=6,
        head_dim=256,
        # レイヤーパターンを定義
        layer_configs=None,
        # 学習時もCacheを使う
        use_cache_in_training=True,
        **kwargs
    ):
        # 例:
        # layer_configs = [
        #     {'type': 'attention', 'num_heads': 32},
        #     {'type': 'gated_deltanet', 'num_heads': 6, 'head_dim': 256},
        #     {'type': 'gated_deltanet', 'num_heads': 6, 'head_dim': 256},
        #     {'type': 'attention', 'num_heads': 32},
        # ]
        self.layer_configs = layer_configs or [
            {'type': 'gated_deltanet'} for _ in range(12)
        ]
        self.use_cache_in_training = use_cache_in_training
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

class FlexibleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            UnifiedMixerBlock(
                config=config,
                layer_idx=i,
                mixer_type=config.layer_configs[i]['type'],
                force_cache_in_training=config.use_cache_in_training
            )
            for i in range(len(config.layer_configs))
        ])

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        **kwargs
    ):
        # Cacheの初期化
        if past_key_values is None and use_cache:
            past_key_values = Cache()

        hidden_states = self.embeddings(input_ids)

        for layer in self.layers:
            hidden_states, _, past_key_values = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )

        return hidden_states, past_key_values
```

### Mamba2との統一のためのCacheアダプター

Mamba2は独自の `Mamba2Cache` を使うため、アダプターが必要です:

```python
class CacheAdapter:
    """FLACacheとMamba2Cacheを統一的に扱うアダプター"""

    def __init__(self, config, batch_size, device, dtype):
        self.fla_cache = Cache()
        self.mamba2_cache = None  # 必要に応じて初期化
        self.layer_types = {}  # {layer_idx: 'fla' or 'mamba2'}

    def register_layer(self, layer_idx, layer_type):
        self.layer_types[layer_idx] = layer_type

        if layer_type == 'mamba2' and self.mamba2_cache is None:
            from fla.models.mamba2.modeling_mamba2 import Mamba2Cache
            self.mamba2_cache = Mamba2Cache(
                config=config,
                batch_size=batch_size,
                dtype=dtype,
                device=device
            )

    def get_cache_for_layer(self, layer_idx):
        layer_type = self.layer_types.get(layer_idx, 'fla')
        if layer_type == 'mamba2':
            return self.mamba2_cache
        else:
            return self.fla_cache
```

### エピソード単位の学習（強化学習など）

```python
model = FlexibleModel(config)
optimizer = torch.optim.AdamW(model.parameters())

# エピソード開始
cache = Cache()  # エピソード全体で共有
episode_data = load_episode()  # タイムステップごとのデータ

for timestep_batch in episode_data:
    # 前のタイムステップの状態を引き継ぐ
    hidden_states, cache = model(
        input_ids=timestep_batch,
        past_key_values=cache,
        use_cache=True  # 学習時でも明示的にTrue
    )

    loss = compute_loss(hidden_states, targets)
    loss.backward()

    # 勾配は計算されるが、Cacheは保持される
    optimizer.step()
    optimizer.zero_grad()

# エピソード終了時にCacheをリセット
cache = Cache()  # 新しいエピソード用
```

---

## まとめ

### 実装のベストプラクティス

1. **統一インターフェース**: すべてのレイヤーが `(hidden_states, attentions, past_key_values)` を返す
2. **学習時のCache使用**: `use_cache=True` を明示的に渡す
3. **レイヤータイプの柔軟な切り替え**: Configベースで制御
4. **Attentionの扱い**: KVキャッシュのみだが、インターフェースは同じ
5. **Mamba2との統合**: アダプターパターンで統一

### 注意点

- **メモリ使用量**: Cacheを保持すると学習時のメモリ使用量が増加
- **勾配の伝播**: Cacheは勾配計算に影響しない（detachされている）
- **chunkモード**: GatedDeltaNetなどは学習時に`chunk`モードが必須
- **window_size**: Attentionで長時間のキャッシュが不要な場合は`window_size`を設定

この設計により、Attention、GatedDeltaNet、Mamba2などを混在させながら、学習時・推論時の両方で継続的なデータ処理が可能になります。

---

## 参考資料

### 主要ファイル

- **レイヤー実装**:
  - `fla/layers/gated_deltanet.py`
  - `fla/layers/attn.py`
  - `fla/layers/mamba2.py`

- **モデル実装**:
  - `fla/models/gated_deltanet/modeling_gated_deltanet.py`
  - `fla/models/gated_deltanet/configuration_gated_deltanet.py`
  - `fla/models/transformer/modeling_transformer.py`

- **Cache実装**:
  - `fla/models/utils.py` (FLACache, Cache)
  - `fla/models/mamba2/modeling_mamba2.py` (Mamba2Cache)

### 関連論文

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- Mamba2: State Space Models
- Flash Attention
