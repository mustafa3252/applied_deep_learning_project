# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework

# Please see gradcam_example.py for the full code usage
import torch
import numpy as np

# Adversarial Erasing 
def erase_top_region(img, cam, erase_ratio=0.2):
    """
    Erase the most highly activated region in the image, force CAM to look elsewhere.
    """
    cam_np = cam.detach().cpu().numpy()[0, 0]
    flat_cam = cam_np.flatten()
    threshold = np.percentile(flat_cam, 100 - erase_ratio * 100)

    erase_mask = (cam_np >= threshold).astype(np.float32)
    erase_mask = torch.tensor(erase_mask, device=img.device).unsqueeze(0).unsqueeze(0)

    erased_img = img.clone()
    erased_img = erased_img * (1 - erase_mask)
    return erased_img

# Consistency Regularization
def horizontal_flip(tensor):
    """
    Flip a 4D tensor horizontally (B, C, H, W)
    """
    return torch.flip(tensor, dims=[3])

"""
# Example Usage

# 1. Adversarial Erasing
erased_img = erase_top_region(img, cam1, erase_ratio=0.2)
erased_img = erased_img.detach().requires_grad_(True)
layer4_cam2 = layer4_gradcam(erased_img, true_class)[0]
layer3_cam2 = layer3_gradcam(erased_img, true_class)[0]
cam1 = layer4_cam2 * 0.7 + layer3_cam2 * 0.3

# 2. Consistency Regularization
flipped_img = horizontal_flip(img).detach().requires_grad_(True)
layer4_cam3 = layer4_gradcam(flipped_img, true_class)[0]
layer3_cam3 = layer3_gradcam(flipped_img, true_class)[0]
cam3 = layer4_cam3 * 0.7 + layer3_cam3 * 0.3
cam2 = horizontal_flip(cam3)

cam = torch.max(torch.max(cam1, cam2)
"""