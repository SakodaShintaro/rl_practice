import torch
from torch import nn

class SmolVLABackbone(nn.Module):
    def __init__(self, model_id: str = "lerobot/smolvla_base") -> None:
        super().__init__()
        from lerobot.common.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

        self.vlm = SmolVLMWithExpertModel(model_id=model_id, load_vlm_weights=True)
        self.processor = self.vlm.processor
        self.hidden_dim = self.vlm.config.text_config.hidden_size

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        batch = [img.cpu() for img in images]
        processed = self.processor(images=batch, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(images.device)
        emb = self.vlm.embed_image(pixel_values)
        return emb.mean(dim=1)
