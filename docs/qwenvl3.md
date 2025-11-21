# QwenVL3をLoRAで学習する

ざっくりいうと

1. Qwen3-VL を LoRA 付きでロード
2. `processor.apply_chat_template` で「テキスト＋画像列」のマルチバッチをまとめてテンソル化
3. `output_hidden_states=True` で Transformer の中間表現を取り出す
4. その中間表現を自前ヘッドに入れて損失を計算

という形になります。

注意点として、現時点では `Qwen/Qwen3-VL-2B-Thinking-FP8` の FP8 重みは Transformers から直接ロードできない、とモデルカードに明記されています（vLLM / SGLang 用）([Hugging Face][1])
なので「実際に PyTorch+Transformers+LoRA で学習する」なら BF16 版の `Qwen/Qwen3-VL-2B-Thinking` を使う前提でコードを書きます（FP8 版でも構造は同じで、ロード部分だけ差し替えるイメージです）。

サンプルとして「画像系列＋プロンプトから 2 クラス分類を学習する」コードを書きます。
画像列は `[img_t0, img_t1, ...]` を 1 サンプルとして扱い、バッチはそれのリストという想定です。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen3-VL-2B-Thinking"
# NOTE:
#   FP8 版 (Qwen/Qwen3-VL-2B-Thinking-FP8) は現状 Transformers から
#   直接ロード不可なので、実運用では BF16 / FP16 版を使ってください。


def build_backbone_with_lora(model_name: str) -> Qwen3VLForConditionalGeneration:
    # Qwen3-VL 本体をロード
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # 環境に応じて
    )

    # Qwen 系でよく使われる LoRA のターゲット例
    # 実際には `print(base_model)` してモジュール名を確認して調整してください
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


class SimpleHead(nn.Module):
    """
    Qwen3-VL の隠れ状態 [B, L, D] から 2 クラス分類をする単純ヘッド
    """

    def __init__(self, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, D]
        return self.net(features)


def prepare_batch(
    processor: AutoProcessor,
    batch_samples,
):
    """
    batch_samples: 長さ B のリスト
        それぞれの要素は
        {
            "images": [PIL.Image または path のリスト],  # 画像系列
            "prompt": str,
            "label": int (0 〜 num_labels-1)
        }
    を想定
    """

    conversations = []
    labels = []

    for sample in batch_samples:
        images = sample["images"]        # list of PIL.Image or paths
        prompt = sample["prompt"]
        label = sample["label"]

        # Qwen2-VL ドキュメントと同じ形式で conversation を組む:contentReference[oaicite:1]{index=1}
        content = []

        # 画像系列をそのまま順番に並べる（時系列に応じてラベルテキスト追加なども可）
        for img in images:
            content.append(
                {
                    "type": "image",
                    # ローカルパスでも PIL.Image でも OK
                    # Transformers の Qwen3VLProcessor が解釈してくれる
                    "image": img,
                }
            )

        content.append({"type": "text", "text": prompt})

        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        conversations.append(conversation)
        labels.append(label)

    # apply_chat_template は Qwen2-VL と同様にバッチの conversation のリストを受け取れる
    # Qwen3-VL でも同じスタイルが採用されています:contentReference[oaicite:2]{index=2}
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=False,  # 学習用なので余計な「応答してください」トークンは付けない
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Qwen 系は token_type_ids が返ることがありますが、学習には不要なので消しておく
    inputs.pop("token_type_ids", None)

    # GPU へ
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    return inputs, labels


def extract_sequence_feature(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "last_non_pad",
) -> torch.Tensor:
    """
    hidden_states: [B, L, D]
    attention_mask: [B, L]
    戻り値: [B, D]
    """
    if mode == "cls":
        # 先頭トークンを特徴として使う（Qwen は専用 CLS がないのであまりおすすめではない）
        return hidden_states[:, 0]

    # デフォルト: パディング以外の最後のトークンを使う
    seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    features = hidden_states[batch_indices, seq_lengths]  # [B, D]
    return features


def training_step(
    backbone: Qwen3VLForConditionalGeneration,
    head: nn.Module,
    processor: AutoProcessor,
    optimizer: torch.optim.Optimizer,
    batch_samples,
):
    backbone.train()
    head.train()

    # 1. プロンプト＋画像系列をまとめてテンソル化（マルチバッチ）
    inputs, labels = prepare_batch(processor, batch_samples)

    # 2. Qwen3-VL を forward。中間表現を取り出したいので hidden_states を出す
    outputs = backbone(
        **inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # outputs.hidden_states は長さ num_layers+1 のタプル
    #   0 番目: embedding 出力
    #  -1 番目: 最終層出力
    hidden_last = outputs.hidden_states[-1]  # [B, L, D]

    # もし視覚トークンだけを使いたい場合は、visual_pos_masks を使って絞ることもできる
    #   visual_pos_masks: [B, L] (視覚トークン位置が 1)
    # ここでは簡単のために「テキスト＋画像すべてを通した最終トークン」を使う
    attention_mask = inputs["attention_mask"]
    features = extract_sequence_feature(hidden_last, attention_mask)  # [B, D]

    # 3. 自前ヘッドに入れてロジットを計算
    logits = head(features)  # [B, num_labels]

    # 4. 任意の損失を計算（ここでは分類用 CrossEntropy）
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    # モデルと Processor, ヘッドを準備
    backbone = build_backbone_with_lora(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    hidden_size = backbone.config.text_config.hidden_size
    head = SimpleHead(hidden_size=hidden_size, num_labels=2).to(DEVICE)

    # Optimizer（LoRA パラメータとヘッドのみ更新）
    # PEFT のモデルは trainable_parameters() だけに gradient が付いているので、
    # 単純に backbone.parameters() を渡しても OK
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )

    # ダミーバッチ（実際には自前データローダから取ってくる）
    from PIL import Image

    dummy_img = Image.new("RGB", (512, 512), color=(128, 128, 128))

    batch_samples = [
        {
            "images": [dummy_img, dummy_img],  # 2 フレームの画像列
            "prompt": "Frame 1 と Frame 2 の違いを説明してください。",
            "label": 0,
        },
        {
            "images": [dummy_img, dummy_img],
            "prompt": "この 2 枚の画像から危険な状況を検出できますか？",
            "label": 1,
        },
    ]

    loss = training_step(
        backbone=backbone,
        head=head,
        processor=processor,
        optimizer=optimizer,
        batch_samples=batch_samples,
    )

    print(f"loss = {loss:.4f}")


if __name__ == "__main__":
    main()
```

ポイントだけ補足します。

* マルチバッチ＋画像系列
  `conversations = [conv1, conv2, ...]` の形で、各 `conv` の `content` に `{"type": "image", "image": img_t}` を時系列順に並べています。
  Qwen2-VL の「Batch Mixed Media Inference」と同じインターフェースで、Qwen3-VL も `processor.apply_chat_template` からマルチバッチを作れます([Hugging Face][2])

* 中間表現の取り方
  `output_hidden_states=True` を付けると `outputs.hidden_states` に各層の `[B, L, D]` が入るので、
  ここでは「パディング以外の最後のトークン」を代表ベクトルとして `SimpleHead` に渡しています。
  画像トークンだけ平均したいなら `inputs["visual_pos_masks"]` でマスクして平均を取るように書き換えれば、画像系列専用の特徴ベクトルにもできます。

* LoRA
  `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` は Qwen 系 Transformer でよく使う設定例です。
  実際には `print(backbone)` して、Attention ブロックの中で LoRA を当てたい線形層名に合わせて調整してください。

この骨組みをベースに、
・ラベルが連続値なら `MSELoss` に変える
・中間層を `hidden_states[layer_index]` から取り出す
・画像トークンだけプーリングする
など、目的に応じて差し替えれば、そのまま「中間表現を別 NN ヘッドで学習する」パターンに使えると思います。

[1]: https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-FP8 "Qwen/Qwen3-VL-2B-Thinking-FP8 · Hugging Face"
[2]: https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl "Qwen2-VL"
