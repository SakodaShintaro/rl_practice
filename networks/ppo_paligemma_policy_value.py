import torch
from torch import nn
from torch.distributions import Beta
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)


class PpoPaligemmaPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        model_id = "google/paligemma2-3b-pt-224"
        self.net = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, cache_dir="./cache"
        )
        self.processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)
        self.prompt = "<image><image> Please drive in the lane."

        seq_hidden_dim = self.net.config.text_config.hidden_size
        rep_dim = 256
        hidden_dim = 100
        self.linear = nn.Linear(seq_hidden_dim, rep_dim)
        self.norm = nn.RMSNorm(rep_dim, elementwise_affine=False)
        self.value_enc = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.ReLU())
        self.value_head = nn.Linear(hidden_dim, 1)
        self.policy_enc = nn.Sequential(nn.Linear(rep_dim, hidden_dim), nn.ReLU())
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        r_seq: torch.Tensor,  # (batch_size, seq_len, 1)
        s_seq: torch.Tensor,  # (batch_size, seq_len, 3, 96, 96)
        a_seq: torch.Tensor,  # (batch_size, seq_len, action_dim)
        action: torch.Tensor | None = None,
    ) -> tuple:
        device = r_seq.device
        batch_size = r_seq.shape[0]
        seq_len = s_seq.shape[1]

        # Prepare all images for batch processing
        s_seq_np = s_seq.cpu().numpy()
        all_images = []
        texts = []

        for batch_idx in range(batch_size):
            s_list = [s_seq_np[batch_idx][i] for i in range(seq_len)]
            all_images.append(s_list)
            texts.append(self.prompt)

        # Process all batches at once
        model_inputs = (
            self.processor(
                text=texts, images=all_images, return_tensors="pt", do_rescale=False, padding=True
            )
            .to(torch.bfloat16)
            .to(device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.net.forward(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            hidden_states = output["hidden_states"]  # tuple, len = 27
            last_hidden_state = hidden_states[-1]  # (batch_size, input_len, seq_hidden_dim)
            x = last_hidden_state[:, input_len - 1]  # (batch_size, seq_hidden_dim)
        x = x.to(torch.float32)  # Convert to float32 for further processing
        x = self.linear(x)  # (batch_size, rep_dim)
        x = self.norm(x)

        value_x = self.value_enc(x)
        value = self.value_head(value_x)

        policy_x = self.policy_enc(x)
        alpha = self.alpha_head(policy_x).exp() + 1
        beta = self.beta_head(policy_x).exp() + 1

        dist = Beta(alpha, beta)
        if action is None:
            action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=-1)

        return {
            "action": action,
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "value_x": value_x,
            "policy_x": policy_x,
        }


if __name__ == "__main__":
    device = torch.device("cuda")
    model = PpoPaligemmaPolicyAndValue(action_dim=3).to(device)

    # Test with batch size > 1
    print("\nTesting batch size 3:")
    r_seq = torch.rand(3, 2, 1, device=device)
    s_seq = torch.rand(3, 2, 3, 96, 96, device=device)
    a_seq = torch.rand(3, 2, 3, device=device)
    output3x1 = model(r_seq, s_seq, a_seq)

    for b in range(3):
        output1x3 = model(r_seq[b : b + 1], s_seq[b : b + 1], a_seq[b : b + 1])
        print(f"{output1x3['action']=}")
        print(f"{output3x1['action'][b]=}")
