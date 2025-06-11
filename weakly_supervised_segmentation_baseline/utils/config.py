# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
import torch

def get_binary_metrics(device=None):
    """
    Creates standard metrics for binary segmentation evaluation without torchmetrics.
    
    Args:
        device: Device to place the metrics on (not used in this implementation)
    
    Returns:
        Dictionary of metric calculation functions
    """
    def calculate_iou(preds, targets):
        intersection = torch.logical_and(preds == 1, targets == 1).sum().float()
        union = torch.logical_or(preds == 1, targets == 1).sum().float()
        return (intersection / (union + 1e-10)).item()
    
    def calculate_dice(preds, targets):
        intersection = torch.logical_and(preds == 1, targets == 1).sum().float()
        return (2 * intersection / (preds.sum() + targets.sum() + 1e-10)).item()
    
    def calculate_pixel_accuracy(preds, targets):
        correct = (preds == targets).sum().float()
        total = preds.numel()
        return (correct / total).item()
    
    metrics = {
        "IoU": calculate_iou,
        "Dice": calculate_dice,
        "Pixel Accuracy": calculate_pixel_accuracy,
    }
            
    return metrics
