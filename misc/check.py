import timm
import torch

model_names = timm.list_models("resnet*")
print(model_names)

model = timm.create_model("resnet18", pretrained=False, num_classes=0)
print(model)

input_tensor = torch.randn(1, 3, 96, 96)
output = model(input_tensor)
print(output.shape)
