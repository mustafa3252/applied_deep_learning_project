# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .common import log_message

def train_classifier(model, dataloader, optimizer, criterion, device):
    """
    Train a classifier model for one epoch.
    
    Args:
        model: Classification model
        dataloader: DataLoader providing images and labels
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to run training on
        
    Returns:
        tuple: (avg_loss, accuracy) for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    log_every = max(1, len(dataloader) // 10)  # Log 10 times per epoch
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training classifier")):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Log batch progress
        if batch_idx % log_every == 0:
            batch_acc = 100 * predicted.eq(labels).sum().item() / labels.size(0)
            log_message(f"  Batch {batch_idx}/{len(dataloader)}: Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%", also_print=False)
            
            # Print a sample prediction
            if batch_idx % (log_every * 5) == 0:
                for i in range(min(2, images.size(0))):
                    log_message(f"    Sample {i}: True={labels[i].item()}, Pred={predicted[i].item()}", also_print=False)
        
    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy


def train_segmentation(model, dataloader, optimizer, criterion, device):
    """
    Train a segmentation model for one epoch.
    
    Args:
        model: Segmentation model
        dataloader: DataLoader providing images and pseudo-masks
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to run training on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    log_every = max(1, len(dataloader) // 10)  # Log 10 times per epoch
    batch_metrics = {"iou_sum": 0, "batch_count": 0}
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training segmentation")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()  # Convert to long for CrossEntropyLoss
        
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate batch IoU for monitoring (simplified)
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            intersection = torch.logical_and(preds == 1, masks == 1).sum().item()
            union = torch.logical_or(preds == 1, masks == 1).sum().item()
            iou = intersection / (union + 1e-10)
            batch_metrics["iou_sum"] += iou
            batch_metrics["batch_count"] += 1
        
        # Log batch progress
        if batch_idx % log_every == 0:
            avg_iou = batch_metrics["iou_sum"] / max(1, batch_metrics["batch_count"])
            log_message(f"  Batch {batch_idx}/{len(dataloader)}: Loss: {loss.item():.4f}, IoU: {avg_iou:.4f}", also_print=False)
            
            # Reset batch metrics
            batch_metrics = {"iou_sum": 0, "batch_count": 0}
        
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate_segmentation(model, dataloader, metrics, device, verbose=False):
    """
    Evaluate segmentation model performance.
    
    Args:
        model: Segmentation model
        dataloader: DataLoader providing images, masks and optionally labels
        metrics: Dictionary of torchmetrics objects
        device: Device to run evaluation on
        verbose: Whether to print detailed metrics
        
    Returns:
        dict: Dictionary of metric names and values
    """
    model.eval()
    # Reset metrics
    for metric in metrics.values():
        metric.reset()
    
    all_ious = []
    class_correct = 0
    class_total = 0
               
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)  # Binary masks (0=background, 1=foreground)
        labels = batch["label"].to(device) if "label" in batch else None
        
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate per-image IoU for logging
        for i in range(images.size(0)):
            pred = preds[i]
            mask = masks[i]
            intersection = torch.logical_and(pred == 1, mask == 1).sum().item()
            union = torch.logical_or(pred == 1, mask == 1).sum().item()
            iou = intersection / (union + 1e-10)
            all_ious.append(iou)
            
            # Log worst and best cases if verbose
            if verbose and (len(all_ious) <= 5 or iou < 0.3 or iou > 0.8):
                log_message(f"  Image {batch_idx * dataloader.batch_size + i}: IoU = {iou:.4f}")
        
        # Track class prediction accuracy if labels are available
        if labels is not None:
            # Simple majority voting to get image-level class
            batch_size, _, h, w = outputs.shape
            for i in range(batch_size):
                # Get the class with highest average response
                cls_response = outputs[i].mean(dim=[1, 2])
                pred_cls = torch.argmax(cls_response).item()
                true_cls = labels[i].item()
                
                class_correct += (pred_cls == true_cls)
                class_total += 1
                
                if verbose and batch_idx < 5:
                    log_message(f"  Image {i}: True class={true_cls}, Pred class={pred_cls}", also_print=False)
        
        # Update metrics
        for name, metric in metrics.items():
            if name == "Dice":
                # Convert to one-hot for Dice score
                preds_onehot = F.one_hot(preds, num_classes=2).permute(0, 3, 1, 2).float()
                masks_onehot = F.one_hot(masks.long(), num_classes=2).permute(0, 3, 1, 2).float()
                metric.update(preds_onehot, masks_onehot)
            elif name == "Pixel Accuracy":
                metric.update(preds.flatten(), masks.long().flatten())
            else: 
                metric.update(preds, masks)
    
    # Log IoU statistics
    all_ious = sorted(all_ious)
    metrics_result = {name: metric.compute().item() for name, metric in metrics.items()}
    
    if verbose and all_ious:
        log_message("\nEvaluation Statistics:")
        log_message(f"  IoU: min={min(all_ious):.4f}, max={max(all_ious):.4f}, median={all_ious[len(all_ious)//2]:.4f}")
        log_message(f"  Bottom 10% IoU: {sum(all_ious[:len(all_ious)//10])/max(1, len(all_ious)//10):.4f}")
        log_message(f"  Top 10% IoU: {sum(all_ious[-len(all_ious)//10:])/max(1, len(all_ious)//10):.4f}")
        
        if class_total > 0:
            log_message(f"  Class prediction accuracy: {100 * class_correct / class_total:.2f}%")
    
    return metrics_result

