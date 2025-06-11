# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework

import torch
import torch.nn.functional as F
from tqdm import tqdm
from .common import log_message


def train_byol(encoder, dataloader, optimizer, device, epochs=20,momentum=0.996):
    """
    Self-supervised BYOL pre-training without external libraries.
    Assumes dataloader yields dicts with two augmented views per sample: 'view1' and 'view2'.

    Args:
        encoder: Backbone model (e.g., ResNet) with an attribute `fc.in_features` for projection dim
        dataloader: DataLoader yielding {'view1': img1, 'view2': img2}
        optimizer: Optimizer for encoder + predictor
        device: 'cpu' or 'cuda'
        epochs: Number of pre-training epochs
        momentum: EMA momentum for target network update

    Returns:
        The encoder with updated weights after BYOL pre-training
    """
    import copy

    target_encoder = copy.deepcopy(encoder).to(device)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    proj_dim = 2048
    predictor = torch.nn.Sequential(
        torch.nn.Linear(proj_dim, proj_dim),
        torch.nn.BatchNorm1d(proj_dim),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(proj_dim, proj_dim)
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    encoder.train()
    predictor.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"BYOL Epoch {epoch}/{epochs}"):
            x1 = batch['view1'].to(device)
            x2 = batch['view2'].to(device)

            out1 = encoder(x1)
            # Flatten if spatial
            if out1.dim() > 2:
                out1 = torch.flatten(out1, start_dim=1)
            pred1 = predictor(out1)

            with torch.no_grad():
                out2 = target_encoder(x2)
                if out2.dim() > 2:
                    out2 = torch.flatten(out2, start_dim=1)
                target_proj = F.normalize(out2, dim=1)

            loss = loss_fn(
                F.normalize(pred1, dim=1),
                target_proj
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            for param_o, param_t in zip(encoder.parameters(), target_encoder.parameters()):
                param_t.data = momentum * param_t.data + (1.0 - momentum) * param_o.data

        avg_loss = total_loss / len(dataloader)
        log_message(f"BYOL Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    return encoder


def train_classifier(model, dataloader, optimizer, criterion, device):
    """
    Train a classifier model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    log_every = max(1, len(dataloader) // 10)

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

        if batch_idx % log_every == 0:
            batch_acc = 100 * predicted.eq(labels).sum().item() / labels.size(0)
            log_message(f"  Batch {batch_idx}/{len(dataloader)}: Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%", also_print=False)

    accuracy = 100 * correct / total
    return running_loss / len(dataloader), accuracy

def train_segmentation(model, dataloader, optimizer, criterion, device):
    """
    Train a segmentation model for one epoch.
    """
    model.train()
    running_loss = 0.0
    log_every = max(1, len(dataloader) // 10)
    iou_accum = 0.0
    iou_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training segmentation")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).long()

        optimizer.zero_grad()
        outputs = model(images)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            intersection = torch.logical_and(preds == 1, masks == 1).sum().item()
            union = torch.logical_or(preds == 1, masks == 1).sum().item()
            iou = intersection / (union + 1e-10)
            iou_accum += iou
            iou_count += 1

        if batch_idx % log_every == 0:
            avg_iou = iou_accum / max(iou_count, 1)
            log_message(f"  Batch {batch_idx}/{len(dataloader)}: Loss: {loss.item():.4f}, IoU: {avg_iou:.4f}" , also_print=False)
            iou_accum = 0.0
            iou_count = 0

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate_segmentation(model, dataloader, metrics, device, verbose=False):
    """
    Evaluate segmentation model performance.
    """
    model.eval()
    for metric in metrics.values():
        metric.reset()

    all_ious = []
    class_correct = 0
    class_total = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(device)

        outputs = model(images)["out"]
        preds = torch.argmax(outputs, dim=1)

        for i in range(images.size(0)):
            pred = preds[i]
            mask = masks[i]
            intersection = torch.logical_and(pred == 1, mask == 1).sum().item()
            union = torch.logical_or(pred == 1, mask == 1).sum().item()
            iou = intersection / (union + 1e-10)
            all_ious.append(iou)
            if verbose and (len(all_ious) <= 5 or iou < 0.3 or iou > 0.8):
                log_message(f"  Image {batch_idx * dataloader.batch_size + i}: IoU = {iou:.4f}")

        if labels is not None:
            for i in range(images.size(0)):
                cls_response = outputs[i].mean(dim=[1, 2])
                pred_cls = torch.argmax(cls_response).item()
                true_cls = labels[i].item()
                class_correct += int(pred_cls == true_cls)
                class_total += 1
                if verbose and batch_idx < 5:
                    log_message(f"  Image {i}: True class={true_cls}, Pred class={pred_cls}" , also_print=False)

        for name, metric in metrics.items():
            n = name.lower()
            if n.startswith("dice"):
                preds_onehot = F.one_hot(preds, num_classes=2).permute(0, 3, 1, 2).float()
                masks_onehot = F.one_hot(masks.long(), num_classes=2).permute(0, 3, 1, 2).float()
                metric.update(preds_onehot, masks_onehot)
            elif n.startswith("pixel accuracy"):
                metric.update(preds.flatten(), masks.long().flatten())
            else:
                metric.update(preds, masks)

    results = {name: m.compute().item() for name, m in metrics.items()}

    if verbose and all_ious:
        sorted_ious = sorted(all_ious)
        half = len(sorted_ious) // 2
        log_message(f"\nIoU stats — min: {sorted_ious[0]:.4f}, max: {sorted_ious[-1]:.4f}, median: {sorted_ious[half]:.4f}")
        if class_total > 0:
            log_message(f"Class Acc: {100 * class_correct / class_total:.2f}%")

    return results