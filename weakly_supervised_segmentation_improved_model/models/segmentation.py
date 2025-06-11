# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def get_segmentation_model(device=None):
    """
    Creates a binary segmentation model based on FCN-ResNet50.

    Args:
        device (torch.device, optional): The device to place the model on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: An FCN-ResNet50 model modified for binary segmentation (foreground vs background).
    """
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=1)  # Update the final layer for binary segmentation
    return model.to(device) if device else model