# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with analysis  of error outputs for debugging

# Gen AI used: ChatGPT

# For the changes in this specific experiment compared to the main code, Gen AI was not used


import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from itertools import product

# Import from local modules
from datasets import (
    OxfordPetClassification, OxfordPetSegmentation,
    PseudoLabeledDataset, BYOLOxfordPetDataset
)
from models import get_classifier, get_segmentation_model, get_encoder
from utils import (
    set_seed, log_message, get_device, get_transforms,
    GradCAM, generate_pseudo_masks,
    train_classifier, train_segmentation, evaluate_segmentation,
    visualize_predictions, visualize_pseudo_masks,
    get_binary_metrics, create_graph, train_byol
)

def main():
    # Setup
    set_seed(42)
    device = get_device()
    transforms = get_transforms()

    os.makedirs("logs", exist_ok=True)
    with open("logs/training_log.txt", "w") as f:
        f.write(f"=== Weakly Supervised Segmentation Pipeline - {datetime.now()} ===\n\n")

    log_message("Starting weakly supervised segmentation pipeline...")
    log_message(f"Device: {device}")
    log_message(f"PyTorch version: {torch.__version__}")

    # STEP 1: Create classification datasets
    log_message("\n1. Setting up classification datasets...")
    train_dataset_cls = OxfordPetClassification(root='./data', split='trainval', transform=transforms['train'])
    val_dataset_cls = OxfordPetClassification(root='./data', split='test', transform=transforms['val'])
    train_size = int(0.8 * len(train_dataset_cls))
    val_size = len(train_dataset_cls) - train_size
    train_dataset_cls, val_dataset_cls_internal = torch.utils.data.random_split(
        train_dataset_cls, [train_size, val_size]
    )
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=16, shuffle=True, num_workers=2)
    val_loader_cls = DataLoader(val_dataset_cls_internal, batch_size=16, shuffle=False, num_workers=2)

    test_dataset_seg = OxfordPetSegmentation(root='./data', split='test', 
        transform_img=transforms['val'], transform_mask=transforms['mask'])
    test_loader_seg = DataLoader(test_dataset_seg, batch_size=16, shuffle=False, num_workers=2)

    log_message(f"Training set (classification): {len(train_dataset_cls)} images")
    log_message(f"Validation set (classification): {len(val_dataset_cls_internal)} images")
    log_message(f"Test set (segmentation): {len(test_dataset_seg)} images")

    # STEP 1.5: BYOL Pre-training
    log_message("\n1.5 Pre-training encoder with BYOL on unlabeled pet images...")
    encoder = get_encoder(device=device)
    byol_dataset = BYOLOxfordPetDataset(root="./data", split="trainval")
    byol_loader = DataLoader(byol_dataset, batch_size=64, shuffle=True, num_workers=2)
    optimizer_byol = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder = train_byol(encoder, byol_loader, optimizer_byol, device, epochs=4, momentum=0.996)
    log_message("BYOL pre-training complete.")

    # STEP 2: Train classifier
    log_message("\n2. Training the classifier...")
    learning_rate = [5e-05]
    weight_decay = [1e-05]
    overall_best_acc = 0
    best_classifier_state = None
    best_params_cls = {}

    for lr, wd in product(learning_rate, weight_decay):
        classifier = get_classifier(device=device, encoder=encoder)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=wd)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(4):
            train_loss, train_acc = train_classifier(classifier, train_loader_cls, optimizer, criterion, device)
            classifier.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader_cls:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    outputs = classifier(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            val_acc = 100 * val_correct / val_total
            log_message(f"Epoch [{epoch+1}/4], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            if val_acc > overall_best_acc:
                overall_best_acc = val_acc
                best_classifier_state = classifier.state_dict()
                best_params_cls = {"lr": lr, "wd": wd}

    log_message(f"Best Classifier settings: {best_params_cls}")
    classifier = get_classifier(device=device, encoder=encoder)
    classifier.load_state_dict(best_classifier_state)
    torch.save(classifier.state_dict(), "models/best_classifier.pth")

    # STEP 3: Generate pseudo-masks
    log_message("\n3. Generating pseudo-masks using GradCAM...")
    full_train_dataset = OxfordPetClassification(root='./data', split='trainval', transform=transforms['train'])
    full_train_loader = DataLoader(full_train_dataset, batch_size=8, shuffle=False, num_workers=2)
    pseudo_labeled_data = generate_pseudo_masks(classifier, full_train_loader, device, threshold=0.5)
    pseudo_dataset = PseudoLabeledDataset(pseudo_labeled_data)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=True, num_workers=2)
    visualize_pseudo_masks(pseudo_dataset, output_dir="gradcam_visualizations")

    # STEP 4: Train segmentation model
    log_message("\n4. Training segmentation model on pseudo-masks...")
    learning_rate_segmentation = [0.0001]
    weight_decay_segmentation = [0.0001]
    best_seg_iou = 0
    best_seg_state = None
    best_seg_params = {}
    tracking_seg = {}

    for lr, wd in product(learning_rate_segmentation, weight_decay_segmentation):
        segmentation_model = get_segmentation_model(device=device)
        optimizer_seg = torch.optim.Adam(segmentation_model.parameters(), lr=lr, weight_decay=wd)
        criterion_seg = torch.nn.CrossEntropyLoss()
        metrics = get_binary_metrics(device)
        train_losses, val_ious = [], []

        for epoch in range(4):
            loss = train_segmentation(segmentation_model, pseudo_loader, optimizer_seg, criterion_seg, device)
            metrics_eval = evaluate_segmentation(segmentation_model, test_loader_seg, metrics, device)
            train_losses.append(loss)
            val_ious.append(metrics_eval["IoU"])
            log_message(f"Epoch [{epoch+1}/4], Train Loss: {loss:.4f}, Val IoU: {metrics_eval['IoU']:.4f}, Dice: {metrics_eval['Dice']:.4f}")

            if metrics_eval['IoU'] > best_seg_iou:
                best_seg_iou = metrics_eval['IoU']
                best_seg_state = segmentation_model.state_dict()
                best_seg_params = {"lr": lr, "wd": wd}

        key = f"lr={lr}, wd={wd}"
        tracking_seg[key] = {"Training Loss": train_losses, "IoU": val_ious}

    log_message(f"Best Segmentation settings: {best_seg_params}")
    segmentation_model = get_segmentation_model(device=device)
    segmentation_model.load_state_dict(best_seg_state)
    torch.save(segmentation_model.state_dict(), "models/best_segmentation_model.pth")
    create_graph(tracking_seg)

    # STEP 5: Final evaluation
    log_message("\n5. Final evaluation on test set...")
    final_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, metrics, device, verbose=True)
    for name, value in final_metrics.items():
        log_message(f"  {name}: {value:.4f}")

    # STEP 6: Visualize predictions
    log_message("\n6. Visualizing results...")
    visualize_predictions(segmentation_model, test_loader_seg, device, grad_cam_enabled=True, num_samples=5)

    log_message("\nPipeline completed!")

if __name__ == "__main__":
    main()
