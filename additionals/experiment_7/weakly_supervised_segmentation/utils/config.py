# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework

import torch
from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import Accuracy

def get_binary_metrics(device=None):
    """
    Creates standard metrics for binary segmentation evaluation.
    
    Args:
        device: Device to place the metrics on
    
    Returns:
        Dictionary of torchmetrics objects
    """
    metrics = {
        "IoU": MeanIoU(num_classes=2),
        "Dice": DiceScore(num_classes=2, average="macro"),
        "Pixel Accuracy": Accuracy(task="binary",num_classes=2, average="macro"),
        }
    
    if device is not None:
        for metric in metrics.values():
            metric = metric.to(device)
            
    return metrics

