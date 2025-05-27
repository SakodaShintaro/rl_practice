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
        # Currently, only supports batch size of 1
        assert r_seq.shape[0] == 1
        s_seq_np = s_seq.cpu().numpy()
        s_list = [s_seq_np[0][i] for i in range(s_seq_np.shape[1])]

        model_inputs = (
            self.processor(text=self.prompt, images=[s_list], return_tensors="pt", do_rescale=False)
            .to(torch.bfloat16)
            .to(device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generation = self.net.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            hidden_states = generation["hidden_states"]  # (ステップ数, レイヤー数)の2層tuple
            last_hidden_state = hidden_states[-1]
            last_layer = last_hidden_state[-1]  # (1, 1, seq_hidden_dim)

            generated_ids = generation["sequences"]
            generated_ids = generated_ids[0][input_len:]  # (出力トークン数, )
            decoded_str = self.processor.decode(generated_ids, skip_special_tokens=True)

        x = last_layer[:, -1]  # Use the last time step representation (batch_size, seq_hidden_dim)
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
        a_logp = dist.log_prob(action).sum(dim=1)

        return {
            "action": action,
            "a_logp": a_logp,
            "value": value,
            "x": x,
            "value_x": value_x,
            "policy_x": policy_x,
            "decoded_str": decoded_str,
        }


if __name__ == "__main__":
    model = PpoPaligemmaPolicyAndValue(action_dim=3)
    r_seq = torch.rand(1, 2, 1)
    s_seq = torch.rand(1, 2, 3, 96, 96)
    a_seq = torch.rand(1, 2, 3)
    output = model(r_seq, s_seq, a_seq)
    print(output)
