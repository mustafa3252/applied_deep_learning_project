# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
 
# Gen AI used: Claude, ChatGPT and Co-Pilot



# segmentation.py
import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

def get_segmentation_model(device=None):
    """
    Creates a binary segmentation model based on FCN-ResNet50.
    
    Args:
        device: The device to place the model on (CPU/CUDA)
    
    Returns:
        A PyTorch FCN-ResNet50 model modified for binary segmentation
    """
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    # Binary segmentation (foreground vs background)
    model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=1)
    return model.to(device) if device else model