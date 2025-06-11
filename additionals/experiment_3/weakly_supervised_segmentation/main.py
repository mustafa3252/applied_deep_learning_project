# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework

import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from itertools import product


from datasets import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset
from models import get_classifier, get_segmentation_model
from utils import (
    set_seed, log_message, get_device, get_transforms,
    GradCAM, generate_pseudo_masks,
    train_classifier, train_segmentation, evaluate_segmentation,
    visualize_predictions, visualize_pseudo_masks,
    get_binary_metrics, create_graph
)

from utils.gradcam import tune_crf_params_optuna


def classifier_training_hypertuning(train_loader_cls, val_loader_cls, device, learning_rate, weight_decay, epochs):
    classifier = get_classifier(device=device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    best_classifier_state = None
    for epoch in range(epochs):
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
        log_message(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_classifier_state = classifier.state_dict()
    return best_val_acc, best_classifier_state


def segmentation_hypertuning_training(pseudo_loader, test_loader_seg, device, learning_rate, weight_decay, epochs):
    track_train_seg_loss = []
    track_iou_seg = []
    segmentation_model = get_segmentation_model(device=device)
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    binary_metrics = get_binary_metrics(device)
    best_val_iou = 0
    best_seg_model_state = None
    for epoch in range(epochs):
        train_loss = train_segmentation(segmentation_model, pseudo_loader, optimizer, criterion, device)
        log_message(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")
        track_iou = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device)
        track_iou_seg.append(track_iou["IoU"])
        log_message(f"Validation IoU: {track_iou['IoU']:.4f}, Dice: {track_iou['Dice']:.4f}")
        if track_iou['IoU'] > best_val_iou:
            best_val_iou = track_iou['IoU']
            best_seg_model_state = segmentation_model.state_dict()
    return best_val_iou, best_seg_model_state, binary_metrics, track_train_seg_loss, track_iou_seg


def main():
    set_seed(42)
    device = get_device()
    transforms = get_transforms()

    os.makedirs("logs", exist_ok=True)
    with open("logs/training_log.txt", "w") as f:
        f.write(f"=== Weakly Supervised Segmentation Pipeline - {datetime.now()} ===\n\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log_message("Starting weakly supervised segmentation pipeline...")
    log_message(f"Device: {device}")
    log_message(f"PyTorch version: {torch.__version__}")

    # Classification datasets
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
    test_dataset_seg = OxfordPetSegmentation(root='./data', split='test', transform_img=transforms['val'], transform_mask=transforms['mask'])
    test_loader_seg = DataLoader(test_dataset_seg, batch_size=16, shuffle=False, num_workers=2)

    log_message(f"Training set (classification): {len(train_dataset_cls)} images")
    log_message(f"Validation set (classification): {len(val_dataset_cls_internal)} images")
    log_message(f"Test set (segmentation): {len(test_dataset_seg)} images")

    # Train classifier
    log_message("\n2. Training the classifier...")
    overall_best_acc = 0
    overall_best_classifier_state = None
    for lr, wd in product([5e-5], [1e-5]):
        acc, state = classifier_training_hypertuning(
            train_loader_cls, val_loader_cls, device, lr, wd, epochs=1
        )
        log_message(f"Validation Accuracy = {acc:.2f}%")
        if acc > overall_best_acc:
            overall_best_acc = acc
            overall_best_classifier_state = state
    log_message(f"Best classifier accuracy: {overall_best_acc:.2f}%")
    os.makedirs("models", exist_ok=True)
    classifier = get_classifier(device=device)
    classifier.load_state_dict(overall_best_classifier_state)
    torch.save(classifier.state_dict(), "models/best_classifier.pth")

    # Tune CRF with Optuna
    log_message("\n3a. Tuning CRF hyperparameters with Optuna...")
    best_crf_params, best_crf_iou = tune_crf_params_optuna(
        classifier,
        test_loader_seg,
        device,
        gt_extractor=lambda b: b['mask'],
        n_trials=10,
        max_batches=30
    )
    log_message(f"Best CRF params: {best_crf_params}, IoU: {best_crf_iou:.4f}")

    # Generate pseudo-masks
    log_message("\n3b. Generating pseudo-masks using GradCAM + CRF...")
    full_train_dataset = OxfordPetClassification(root='./data', split='trainval', transform=transforms['train'])
    full_train_loader = DataLoader(full_train_dataset, batch_size=8, shuffle=False, num_workers=2)
    pseudo_labeled_data = generate_pseudo_masks(
        classifier,
        full_train_loader,
        device,
        save_visualizations=False,
        visualization_dir="gradcam_visualizations",
        crf_params=best_crf_params
    )
    log_message(f"Generated {len(pseudo_labeled_data)} pseudo-labeled samples")

    # Train segmentation model on pseudo-masks
    pseudo_dataset = PseudoLabeledDataset(pseudo_labeled_data)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=True, num_workers=2)
    visualize_pseudo_masks(pseudo_dataset, output_dir="gradcam_visualizations")

    log_message("\n4. Training segmentation model on pseudo-masks...")
    overall_best_segmentation_iou = 0
    overall_best_segmentation_state = None
    tracking_seg = {}
    for lr_seg, wd_seg in product([1e-4], [1e-4]):
        val_iou, seg_state, binary_metrics, train_loss_hist, iou_hist = segmentation_hypertuning_training(
            pseudo_loader, test_loader_seg, device, lr_seg, wd_seg, epochs=1
        )
        params_str = f"lr={lr_seg}, wd={wd_seg}"
        tracking_seg[params_str] = {"Loss": train_loss_hist, "IoU": iou_hist}
        log_message(f"Validation IoU = {val_iou:.2f}%")
        if val_iou > overall_best_segmentation_iou:
            overall_best_segmentation_iou = val_iou
            overall_best_segmentation_state = seg_state
    log_message(f"Best segmentation IoU: {overall_best_segmentation_iou:.2f}%")
    segmentation_model = get_segmentation_model(device=device)
    segmentation_model.load_state_dict(overall_best_segmentation_state)
    torch.save(segmentation_model.state_dict(), "models/best_segmentation_model.pth")
    create_graph(tracking_seg)

    # Final evaluation on test set
    log_message("\n5. Final evaluation on test set...")
    final_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device, verbose=True)
    for name, value in final_metrics.items():
        log_message(f"  {name}: {value:.4f}")

    # Visualize results
    log_message("\n6. Visualizing results...")
    visualize_predictions(segmentation_model, test_loader_seg, device, grad_cam_enabled=True, num_samples=5)

    log_message("\nWeakly supervised segmentation pipeline completed!")


if __name__ == "__main__":
    main()
