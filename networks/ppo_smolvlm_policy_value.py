import torch
from torch import nn
from torch.distributions import Beta
from transformers import AutoModelForVision2Seq, AutoProcessor


class PpoSmolVlmPolicyAndValue(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        model_id = "HuggingFaceTB/SmolVLM-Instruct"
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
        self.net = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
            cache_dir="./cache",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
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
        r_seq: torch.Tensor,
        s_seq: torch.Tensor,
        a_seq: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> dict:
        device = r_seq.device
        batch_size = r_seq.shape[0]
        seq_len = s_seq.shape[1]

        s_seq_np = s_seq.cpu().numpy()
        all_images = []
        texts = []

        for batch_idx in range(batch_size):
            s_list = [s_seq_np[batch_idx][i] for i in range(seq_len)]
            all_images.append(s_list)
            texts.append(self.prompt)

        model_inputs = (
            self.processor(
                text=texts,
                images=all_images,
                return_tensors="pt",
                do_rescale=False,
                padding=True,
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
            hidden_states = output["hidden_states"]
            last_hidden_state = hidden_states[-1]
            x = last_hidden_state[:, input_len - 1]
        x = x.to(torch.float32)
        x = self.linear(x)
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
