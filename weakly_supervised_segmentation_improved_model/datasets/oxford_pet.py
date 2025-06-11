# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from utils.gradcam import flip_mask_pixels

class OxfordPetClassification(Dataset):
    """
    Dataset for image classification (pet breeds) using the Oxford-IIIT Pet dataset.
    """
    def __init__(self, root, split='trainval', transform=None):
        """
        Initialize the classification dataset.

        Args:
            root (str): Path to the dataset directory.
            split (str): Dataset split ('trainval' or 'test').
            transform (callable, optional): Transformations for the images.
        """
        self.dataset = OxfordIIITPet(root, split=split, download=True)
        self.transform = transform
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with 'image' and 'label'.
        """
        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": target}


class OxfordPetSegmentation(Dataset):
    """
    Dataset for segmentation tasks with images and ground truth masks.
    """
    def __init__(self, root, split='test', transform_img=None, transform_mask=None):
        """
        Initialize the segmentation dataset.

        Args:
            root (str): Path to the dataset directory.
            split (str): Dataset split ('trainval' or 'test').
            transform_img (callable, optional): Transformations for the images.
            transform_mask (callable, optional): Transformations for the masks.
        """
        self.dataset = OxfordIIITPet(root, split=split, target_types='segmentation', download=True)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with 'image', 'mask', and 'label'.
        """
        img, mask = self.dataset[idx]
        
        # Get the class label for the image
        category_dataset = OxfordIIITPet(
            root=self.dataset.root, 
            split=self.dataset._split, 
            target_types="category",
            download=False
        )
        _, class_target = category_dataset[idx]
        
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask).squeeze(0).long()

            # Convert trinary mask to binary (foreground vs background)
            binary_mask = torch.zeros_like(mask)
            binary_mask[mask == 1] = 1  # Pet
            binary_mask[mask == 2] = 1  # Include boundary with pet
            
        return {"image": img, "mask": binary_mask, "label": class_target}


class PseudoLabeledDataset(Dataset):
    """
    Dataset for pseudo-labeled data generated using GradCAM.
    """
    def __init__(self, pseudo_labeled_data, noise_ratio=0.0):
        """
        Initialize the pseudo-labeled dataset.

        Args:
            pseudo_labeled_data (list): List of dictionaries with 'image', 'pseudo_mask', and 'true_class'.
            noise_ratio (float): Percentage of pixels to flip in the pseudo-masks (0.0-1.0).
        """
        self.data = pseudo_labeled_data
        self.noise_ratio = noise_ratio
        
        # Apply synthetic noise
        if self.noise_ratio > 0:
            for item in self.data:
                item["pseudo_mask"] = flip_mask_pixels(item["pseudo_mask"], self.noise_ratio)
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with 'image', 'mask', and 'label'.
        """
        return {
            "image": self.data[idx]["image"].squeeze(0),
            "mask": self.data[idx]["pseudo_mask"].squeeze(0).squeeze(0),
            "label": self.data[idx]["true_class"]
        }