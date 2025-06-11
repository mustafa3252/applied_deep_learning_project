# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

import torch
from torchvision.models import resnet50, ResNet50_Weights

def get_classifier(device=None):
    """
    Creates a ResNet50-based classifier for the Oxford-IIIT Pet dataset.

    Args:
        device (torch.device, optional): The device to place the model on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: A ResNet50 model modified for 37 pet breed classes.
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 37)  # Update the final layer for 37 classes
    return model.to(device) if device else model