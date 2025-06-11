# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
 
# Gen AI used: Claude, ChatGPT and Co-Pilot



import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from datetime import datetime
from torch.utils.data import DataLoader

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_message(message, also_print=True):
    """
    Log a message to file and optionally print to console.
    
    Args:
        message: Message to log
        also_print: Whether to also print the message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Append to log file
    with open("logs/training_log.txt", "a") as f:
        f.write(log_msg + "\n")
    
    if also_print:
        print(log_msg)


def get_device():
    """
    Get the device to run on (CPU or CUDA).
    
    Returns:
        torch.device: Device to use
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """
    Get image and mask transforms for training and validation.
    
    Returns:
        dict: Dictionary with train, validation and mask transforms
    """
    # Transform with augmentation for training
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Transform for validation/testing
    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Transform for masks (only used for evaluation)
    transform_mask = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
        T.PILToTensor(),
    ])
    
    return {
        'train': transform_train,
        'val': transform_val,
        'mask': transform_mask
    }