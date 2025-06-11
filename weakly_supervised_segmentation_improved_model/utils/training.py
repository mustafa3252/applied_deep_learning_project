# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot
import torch
import torch.nn.functional as F
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
    print("Training classifier...")

    for batch_idx, batch in enumerate(dataloader):
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
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()
        
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
            batch_metrics = {"iou_sum": 0, "batch_count": 0}
        
    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate_segmentation(model, dataloader, metrics, device, verbose=False):
    model.eval()

    metrics_accum = {name: 0.0 for name in metrics}
    metrics_count = 0
    
    all_ious = []
    class_correct = 0
    class_total = 0
               
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"Evaluating batch {batch_idx}/{len(dataloader)}")
            
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device) if "label" in batch else None
        
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate per-image IoU for logging
        for i in range(images.size(0)):
            pred = preds[i]
            mask = masks[i]
            
            # Update metrics for each image
            for name, metric_fn in metrics.items():
                metric_value = metric_fn(pred, mask)
                metrics_accum[name] += metric_value
            
            # Calculate IoU for this specific image for logging
            intersection = torch.logical_and(pred == 1, mask == 1).sum().item()
            union = torch.logical_or(pred == 1, mask == 1).sum().item()
            iou = intersection / (union + 1e-10)
            all_ious.append(iou)
            
            metrics_count += 1
        
        # Track class prediction accuracy if labels are available
        if labels is not None:
            # Process classification accuracy with PyTorch operations
            batch_size, _, h, w = outputs.shape
            for i in range(batch_size):
                # Get the class with highest average response
                cls_response = outputs[i].mean(dim=[1, 2])
                pred_cls = torch.argmax(cls_response).item()
                true_cls = labels[i].item()
                
                class_correct += (pred_cls == true_cls)
                class_total += 1
    
    # Calculate final metrics
    metrics_result = {name: accum / metrics_count for name, accum in metrics_accum.items()}
    
    return metrics_result