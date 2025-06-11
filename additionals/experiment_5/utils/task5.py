# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework

import torch
import random
from torch.utils.data import Subset, DataLoader

from .visualization import visualize_predictions
from .config import get_binary_metrics
from .training import train_segmentation, evaluate_segmentation
from .common import log_message


from collections import defaultdict
from torch.utils.data import Subset
# This file was adapted for experiment 5. More details can be found below in the code.
# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Chatgpt and co-pilot
def get_fixed_per_class_subset(full_dataset, n_per_class=3):
    label_to_indices = defaultdict(list)

    for i in range(len(full_dataset)):
        item = full_dataset[i]
        class_label = item["label"] if isinstance(item, dict) else item[2]
        label_to_indices[class_label].append(i)

    selected_indices = []
    for cls, indices in label_to_indices.items():
        if len(indices) <= n_per_class:
            selected = indices
        else:
            selected = random.sample(indices, n_per_class)
        selected_indices.extend(selected)

    print(f"Selected total {len(selected_indices)} samples from {len(label_to_indices)} classes "
          f"({n_per_class} per class)")

    return Subset(full_dataset, selected_indices)



def fine_tune_on_real_masks(segmentation_model, full_dataset, test_loader_seg, device,
                            epochs=3, scale=0.1, batch_size=16, learning_rate=2e-6,
                            save_path="models/fine_tuned_segmentation_model.pth"):
    """
    Fine-tune the segmentation model on a subset of real masks.
    Args:
        segmentation_model: Pre-trained segmentation model
        full_dataset: Full dataset containing real masks
        test_loader_seg: DataLoader for validation set
        device: Device to run training on (CPU or CUDA)
        epochs: Number of epochs for fine-tuning
        scale: Fraction of the dataset to use for fine-tuning, default is 10%
        batch_size: Batch size for DataLoader, default is 16
        learning_rate: Learning rate for optimizer
        save_path: Path to save the fine-tuned model
    Returns:
        segmentation_model: Fine-tuned segmentation model
    """
    track_fine_tune_loss = []
    track_iou_fine_tune = []
    binary_metrics = get_binary_metrics(device)
    # Step 1: Create a subset of the dataset
    # total_size = len(full_dataset)
    # subset_size = int(scale * total_size)
    # indices = random.sample(range(total_size), subset_size)
    # fine_tune_dataset = Subset(full_dataset, indices)
    # fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    fine_tune_dataset = get_fixed_per_class_subset(full_dataset)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # Step 2: Optimizer & loss
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # Step 3: Training loop
    log_message(f"\n[Fine-Tune] Starting fine-tuning on {len(full_dataset)} real mask samples ({scale * 100: .1f}%)...")
    segmentation_model.train()
    for epoch in range(epochs):
        # Train for one epoch
        loss = train_segmentation(segmentation_model, fine_tune_loader, optimizer, criterion, device)
        track_fine_tune_loss.append(loss)
        log_message(f"[Fine-Tune Epoch {epoch + 1}/{epochs}] Loss: {loss: .4f}")
        # Validate on test set
        eval_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device)
        track_iou_fine_tune.append(eval_metrics["IoU"])
        log_message(f"Validation IoU: {eval_metrics['IoU']:.4f}, Dice: {eval_metrics['Dice']:.4f}")
        visualize_predictions(segmentation_model, test_loader_seg, device, grad_cam_enabled=True, num_samples=5,
                              name=f'fine_tune_epoch_{epoch + 1}_predictions.png')
    # Step 4: Save fine-tuned model
    torch.save(segmentation_model.state_dict(), save_path)
    log_message(f"[Fine-Tune] Fine-tuned model saved to {save_path}")
    log_message(f"track_fine_tune_loss: {track_fine_tune_loss} \n")
    log_message(f"track_iou_fine_tune: {track_iou_fine_tune} \n")
    return segmentation_model
