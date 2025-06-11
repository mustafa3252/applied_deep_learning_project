# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

# models/encoder.py
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

def get_encoder(pretrained=False, device='cpu'):
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Identity()
    return model.to(device)
