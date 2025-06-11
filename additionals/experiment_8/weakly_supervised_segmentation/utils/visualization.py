# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from .gradcam import GradCAM
from .common import log_message

def visualize_predictions(model, dataloader, device, grad_cam_enabled=True, num_samples=5):
    """
    Visualize model predictions and optionally GradCAM heatmaps.
    
    Args:
        model: Segmentation model
        dataloader: DataLoader providing images and masks
        device: Device for inference
        grad_cam_enabled: Whether to include GradCAM visualizations
        num_samples: Number of samples to visualize
        
    Returns:
        Path to saved visualization image
    """
    model.eval()
    samples = next(iter(dataloader))
    images = samples["image"].to(device)[:num_samples]
    masks = samples["mask"][:num_samples]
    
    with torch.no_grad():
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
    
    # Create output directory
    os.makedirs("visualization_results", exist_ok=True)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # Fixed shape for single image
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)   # Fixed shape for single image
    
    fig, axes = plt.subplots(num_samples, 4 if grad_cam_enabled else 3, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        img = images[i].cpu()
        # Reshape mean and std to match img dimensions (no batch dimension)
        img = (img * std) + mean  # Denormalize
        img = img.permute(1, 2, 0).numpy()  # Now correctly permute [C,H,W] -> [H,W,C]
        img = np.clip(img, 0, 1)
            
        true_mask = masks[i].squeeze().numpy()
        pred_mask = preds[i].cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        
        # GradCAM if enabled
        if grad_cam_enabled:
            # Use the 'layer4' of backbone for GradCAM
            target_layer = model.backbone.layer4[-1]
            grad_cam = GradCAM(model, target_layer)
            cam, target_class = grad_cam(images[i:i+1], target_class=1)  # Focus on foreground
            cam_np = cam.squeeze().cpu().numpy()
                
            # Overlay CAM on original image
            axes[i, 3].imshow(img)
            axes[i, 3].imshow(cam_np, cmap='jet', alpha=0.5)
            axes[i, 3].set_title("GradCAM")
            axes[i, 3].axis('off')
            
            grad_cam.remove_hooks()
    
    plt.tight_layout()
    output_path = "visualization_results/predictions.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    log_message(f"Saved prediction visualizations to {output_path}")
    return output_path


def visualize_pseudo_masks(pseudo_dataset, output_dir="gradcam_visualizations", num_samples=9):
    """
    Visualize pseudo masks generated using GradCAM.
    
    Args:
        pseudo_dataset: Dataset containing pseudo-labeled samples
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize in a grid
    
    Returns:
        Path to saved visualization image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a grid of images with their pseudo-masks
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # Denormalize images
    mean_arr = np.array([0.485, 0.456, 0.406])
    std_arr = np.array([0.229, 0.224, 0.225])
    
    # Choose random samples to visualize
    sample_indices = np.random.choice(len(pseudo_dataset), 
                                      min(num_samples, len(pseudo_dataset)), 
                                      replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = pseudo_dataset[idx]
        img = sample["image"].permute(1, 2, 0).numpy()
        img = img * std_arr.reshape(1, 1, 3) + mean_arr.reshape(1, 1, 3)
        img = np.clip(img, 0, 1)
        
        mask = sample["mask"].numpy()
            
        row, col = i // grid_size, i % grid_size
        axes[row, col].imshow(img)
        axes[row, col].imshow(mask, cmap='jet', alpha=0.5)
        axes[row, col].set_title(f"Class {sample['label']}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = f"{output_dir}/pseudo_masks.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    log_message(f"Saved pseudo-mask visualizations to {output_path}")
    return output_path


def save_individual_pseudo_masks(pseudo_dataset, num_to_save=20, output_dir="gradcam_visualizations/individual"):
    """
    Save individual pseudo-masked images for detailed inspection.
    
    Args:
        pseudo_dataset: Dataset containing pseudo-labeled samples
        num_to_save: Number of individual samples to save
        output_dir: Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Denormalize images
    mean_arr = np.array([0.485, 0.456, 0.406])
    std_arr = np.array([0.229, 0.224, 0.225])
    
    # Choose random samples to visualize
    sample_indices = np.random.choice(len(pseudo_dataset), 
                                     min(num_to_save, len(pseudo_dataset)), 
                                     replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = pseudo_dataset[idx]
        img = sample["image"].permute(1, 2, 0).numpy()
        img = img * std_arr.reshape(1, 1, 3) + mean_arr.reshape(1, 1, 3)
        img = np.clip(img, 0, 1)
        
        mask = sample["mask"].numpy()
        class_id = sample["label"]
        
        # Create figure for this sample
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Pseudo-mask only
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Pseudo-mask")
        axes[1].axis('off')
        
        # Overlay mask on image
        axes[2].imshow(img)
        axes[2].imshow(mask, cmap='jet', alpha=0.5)
        axes[2].set_title(f"Overlay (Class {class_id})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_{idx}_class_{class_id}.png", 
                    bbox_inches='tight', dpi=200)
        plt.close()
    
    log_message(f"Saved {len(sample_indices)} individual pseudo-mask visualizations to {output_dir}")

def create_graph(tracking_seg):
    plt.figure(figsize=(12, 6))
    for parameters,values in tracking_seg.items():
        plt.plot(values["Training Loss"], marker= 'o', label=f"{parameters} Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epochs")
    plt.savefig("Training_Loss_vs_Epochs.png",dpi=300, bbox_inches='tight')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    for parameters,values in tracking_seg.items():
        plt.plot(values["IoU"],  marker= 'o', label=f"{parameters} Mean IoU")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Mean IoU")
    plt.title("Validation Mean IoU for different hyperparameters values")
    plt.savefig("Validation_Mean_IoU_for_different_hyperparameters_values",dpi=300, bbox_inches='tight')
    plt.grid(True)
    plt.show()
    plt.close()