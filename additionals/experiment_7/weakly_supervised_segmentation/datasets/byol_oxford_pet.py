# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

class BYOLOxfordPetDataset(Dataset):
    """
    Dataset that returns two random augmented views per image for BYOL training.
    """
    def __init__(self, root, split='trainval', transform=None):
        self.dataset = OxfordIIITPet(root, split=split, download=True)
        self.augment = transform or transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        view1 = self.augment(image)
        view2 = self.augment(image)
        return {
            "view1": view1,
            "view2": view2
        }
