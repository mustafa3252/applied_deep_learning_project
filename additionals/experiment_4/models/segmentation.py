# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot
# segmentation.py
# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework
import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch import optim
import torch.nn as nn

def get_segmentation_model(device=None):
    """
    Creates a binary segmentation model based on FCN-ResNet50.
    
    Args:
        device: The device to place the model on (CPU/CUDA)
    
    Returns:
        A PyTorch FCN-ResNet50 model modified for binary segmentation
    """
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    model.classifier[4] = nn.Sequential(
        nn.Conv2d(256, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 3, 1)
    )

    return model.to(device) if device else model

