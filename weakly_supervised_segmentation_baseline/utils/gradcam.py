# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
import torch
import torch.nn.functional as F
from .common import log_message
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

"""
Generates Grad-CAM visualizations for a given model and target layer.
Can be used with both classification and segmentation models.
"""

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model (either classifier or segmentation model)
            target_layer: Layer to extract gradients and activations from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
        
    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer.
        """
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def __call__(self, input_tensor, target_class=None):
        """
        Compute GradCAM for the given input tensor.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index to compute GradCAM for.
                          If None, uses the predicted class for classifiers.
        
        Returns:
            tuple: (cam, target_class), where cam is the normalized attention map
                  and target_class is the class that was used
        """
        self.model.eval()
        self.model.zero_grad()
        
        if hasattr(self.model, 'fc'): 
            logits = self.model(input_tensor)
            _, pred_class = torch.max(logits, 1)
            
            if target_class is None:
                target_class = pred_class.item()
                
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            
            logits.backward(gradient=one_hot, retain_graph=True)
            
        else: 
            output = self.model(input_tensor)['out']
            N, C, H, W = output.shape
            
            if target_class is None:
                target_class = torch.argmax(output.mean(dim=[2, 3])).item()
                
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class] = 1
            
            output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) 
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach(), target_class
        
    def remove_hooks(self):
        """
        Remove all hooks to avoid memory leaks.
        """
        for handle in self.hook_handles:
            handle.remove()

def multi_scale_gradcam(model, img, target_class):
    """
    Generate GradCAM from multiple layers and combine them
    """
    layer4_cam = GradCAM(model, model.layer4[-1])(img, target_class)[0]
    layer3_cam = GradCAM(model, model.layer3[-1])(img, target_class)[0]
    
    combined = layer4_cam * 0.7 + layer3_cam * 0.3
    
    return combined

def apply_crf(image, cam, initial_mask=None):
    img_np = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    img_np = np.ascontiguousarray(img_np * 255).astype(np.uint8)
    cam_np = cam.detach().cpu().numpy()[0, 0]

    if initial_mask is not None:
        initial_np = initial_mask.cpu().numpy()[0, 0]
        prob = np.stack([1-initial_np, initial_np], axis=0) * 0.5 + \
               np.stack([1-cam_np, cam_np], axis=0) * 0.5
    else:
        prob = np.stack([1-cam_np, cam_np], axis=0)

    d = dcrf.DenseCRF2D(img_np.shape[1], img_np.shape[0], 2)
    d.setUnaryEnergy(unary_from_softmax(prob))
    d.addPairwiseGaussian(sxy=2, compat=4)
    d.addPairwiseBilateral(sxy=60, srgb=10, rgbim=img_np, compat=12)

    q = d.inference(5)
    mask = np.argmax(q, axis=0).reshape(cam_np.shape)
    return torch.tensor(mask, device=cam.device).float().unsqueeze(0).unsqueeze(0)


def contour_refinement(binary_mask):
    """
    Refine mask using contour analysis
    """
    mask_np = binary_mask.cpu().numpy()[0, 0].astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    refined = np.zeros_like(mask_np)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(refined, [largest], 0, 1, -1)
    
    return torch.tensor(refined, device=binary_mask.device).float().unsqueeze(0).unsqueeze(0)

def save_gradcam_visualization(image, cam, pseudo_mask, save_path):
    """
    Save visualization with original image, GradCAM heatmap, and resulting mask
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    img = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    
    heatmap = cam.detach().cpu().numpy()[0, 0]
    mask = pseudo_mask.detach().cpu().numpy()[0, 0]
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    overlay = 0.7 * img + 0.3 * heatmap_colored
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("GradCAM Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def percentile_threshold(cam, percentile=80):
    """
    Threshold CAM to keep top N% of activation values
    """
    cam_np = cam.detach().cpu().numpy()[0, 0]
    thresh = np.percentile(cam_np, percentile)
    return (cam > thresh).float()

def generate_pseudo_masks(classifier, dataloader, device, threshold=0.5, save_visualizations=False, visualization_dir="gradcam_visualizations"):
    """
    Generate pseudo segmentation masks using GradCAM from a trained classifier.
    """
    classifier.eval()
    pseudo_labeled_data = []
    
    layer4_gradcam = GradCAM(classifier, classifier.layer4[-1])
    layer3_gradcam = GradCAM(classifier, classifier.layer3[-1])
    
    total_masks = 0
    foreground_pixels = 0
    total_pixels = 0
    class_mask_stats = {}
    
    if save_visualizations:
        os.makedirs(visualization_dir, exist_ok=True)
        log_message(f"Saving GradCAM visualizations to {visualization_dir}")
    
    log_message("Starting pseudo-mask generation...")
    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 5 == 0:
            log_message(f"Generating pseudo-masks: batch {batch_idx}/{total_batches}")
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        for i in range(images.size(0)):
            img = images[i:i+1].clone().detach().requires_grad_(True)
            true_class = labels[i].item()

            layer4_cam = layer4_gradcam(img, true_class)[0]
            layer3_cam = layer3_gradcam(img, true_class)[0]
            cam = layer4_cam * 0.7 + layer3_cam * 0.3
            
            pseudo_mask = percentile_threshold(cam, percentile=80)

            min_foreground_pixels = 0.01 * pseudo_mask.numel()  # At least 1% pixels
            if pseudo_mask.sum() < min_foreground_pixels:
                for fallback_percentile in [70, 60, 50]:
                    pseudo_mask = percentile_threshold(cam, fallback_percentile)
                    if pseudo_mask.sum() >= min_foreground_pixels:
                        break

            if pseudo_mask.sum() == 0:
                pseudo_mask = (cam > cam.mean()).float()

            pseudo_mask = apply_crf(img, cam, initial_mask=pseudo_mask)
            pseudo_mask = contour_refinement(pseudo_mask)
            
            if save_visualizations:
                save_path = os.path.join(visualization_dir, f"sample_{total_masks}_class_{true_class}.png")
                save_gradcam_visualization(images[i:i+1], cam, pseudo_mask, save_path)
            
            total_masks += 1
            mask_foreground = pseudo_mask.sum().item()
            mask_total = pseudo_mask.numel()
            foreground_pixels += mask_foreground
            total_pixels += mask_total
            
            if true_class not in class_mask_stats:
                class_mask_stats[true_class] = {"count": 0, "foreground": 0, "total": 0}
            class_mask_stats[true_class]["count"] += 1
            class_mask_stats[true_class]["foreground"] += mask_foreground
            class_mask_stats[true_class]["total"] += mask_total
            
            if total_masks % 100 == 0:
                coverage = 100 * (foreground_pixels / total_pixels)
                log_message(f"  Generated {total_masks} masks. Avg coverage: {coverage:.2f}%", also_print=False)
            
            pseudo_labeled_data.append({
                "image": images[i:i+1].cpu().detach(),
                "pseudo_mask": pseudo_mask.cpu(),
                "true_class": true_class
            })
    
    log_message(f"\nPseudo-mask generation complete:")
    log_message(f"  Total masks generated: {total_masks}")
    log_message(f"  Average foreground coverage: {100 * (foreground_pixels / total_pixels):.2f}%")
    log_message(f"  Class-wise statistics:")
    for cls, stats in class_mask_stats.items():
        cls_coverage = 100 * (stats["foreground"] / stats["total"])
        log_message(f"Class {cls}: {stats['count']} masks, {cls_coverage:.2f}% avg coverage")
    
    layer4_gradcam.remove_hooks()
    layer3_gradcam.remove_hooks()
    
    return pseudo_labeled_data

