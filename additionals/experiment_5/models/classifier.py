# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework

# classifier.py
import torch
from torchvision.models import resnet50, ResNet50_Weights

def get_classifier(device=None):
    """
    Creates a classifier model based on ResNet50 for the Oxford-IIIT Pet dataset.
    
    Args:
        device: The device to place the model on (CPU/CUDA)
    
    Returns:
        A PyTorch ResNet50 model modified for pet breed classification
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Oxford-IIIT Pet has 37 classes
    model.fc = torch.nn.Linear(model.fc.in_features, 37)
    return model.to(device) if device else model