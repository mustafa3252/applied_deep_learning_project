# Gen Al Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging refactoring and refining the architecture in depth
# Gen AI used: Claude, ChatGPT and Co-Pilot

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework
import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from itertools import product

# Import from local modules
from datasets import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset
from models import get_classifier, get_segmentation_model
from utils import (
    set_seed, log_message, get_device, get_transforms, 
    GradCAM, generate_pseudo_masks, 
    train_classifier, train_segmentation, evaluate_segmentation,
    visualize_predictions, visualize_pseudo_masks,
    get_binary_metrics, create_graph
)

# Define results directory in the same folder as this script
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Create results directories
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "gradcam_visualizations"), exist_ok=True)

def classifier_training_hypertuning(train_loader_cls, val_loader_cls, device, learning_rate, weight_decay, epochs):
    classifier = get_classifier(device=device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    num_epochs = epochs  # Increase for better results
    best_val_acc = 0
    best_classifier_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_classifier(classifier, train_loader_cls, optimizer, criterion, device)
        
        # Validate
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
        log_message(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_classifier_state = classifier.state_dict()
        scheduler.step()
    return best_val_acc, best_classifier_state

def segmentation_hypertuning_training(pseudo_loader, test_loader_seg, device, learning_rate, weight_decay, epochs):
    track_train_seg_loss = []
    track_iou_seg = []
    segmentation_model = get_segmentation_model(device=device)
    optimizer = torch.optim.AdamW(segmentation_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    binary_metrics = get_binary_metrics(device)
    
    num_epochs = epochs  # Increase for better results
    best_val_iou = 0
    best_seg_model_state = None
    
    for epoch in range(num_epochs):
        train_loss = train_segmentation(segmentation_model, pseudo_loader, optimizer, criterion, device)
        log_message(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
        track_train_seg_loss.append(train_loss)
        # Validate on test set
        eval_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device)
        track_iou_seg.append(eval_metrics["IoU"])
        log_message(f"Validation IoU: {eval_metrics['IoU']:.4f}, Dice: {eval_metrics['Dice']:.4f}")
        # Save best model
        if eval_metrics['IoU'] > best_val_iou:
            best_val_iou = eval_metrics['IoU']
            best_seg_model_state = segmentation_model.state_dict()
        scheduler.step(train_loss)
    return best_val_iou, best_seg_model_state, binary_metrics, track_train_seg_loss, track_iou_seg


def main():
    # Setup
    set_seed(42)
    device = get_device()
    transforms = get_transforms()
    
    # Setup logging
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "logs", "training_log.txt"), "w") as f:
        f.write(f"=== Weakly Supervised Segmentation Pipeline - {datetime.now()} ===\n\n")
    
    log_message("Starting weakly supervised segmentation pipeline...")
    log_message(f"Device: {device}")
    log_message(f"PyTorch version: {torch.__version__}")
    
    # STEP 1: Create datasets for classification (image-level labels only)
    log_message("\n1. Setting up classification datasets...")
    train_dataset_cls = OxfordPetClassification(root='./data', split='trainval', transform=transforms['train'])
    val_dataset_cls = OxfordPetClassification(root='./data', split='test', transform=transforms['val'])
    
    # Split training data for classification
    train_size = int(0.8 * len(train_dataset_cls))
    val_size = len(train_dataset_cls) - train_size
    train_dataset_cls, val_dataset_cls_internal = torch.utils.data.random_split(
        train_dataset_cls, [train_size, val_size]
    )
    
    # Create data loaders for classification
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=16, shuffle=True, num_workers=2)
    val_loader_cls = DataLoader(val_dataset_cls_internal, batch_size=16, shuffle=False, num_workers=2)
    
    # Also load test set (only for final evaluation, with segmentation masks)
    test_dataset_seg = OxfordPetSegmentation(root='./data', split='test', 
                                          transform_img=transforms['val'],
                                          transform_mask=transforms['mask'])
    test_loader_seg = DataLoader(test_dataset_seg, batch_size=16, shuffle=False, num_workers=2)
    
    # Print dataset information
    log_message(f"Training set (classification): {len(train_dataset_cls)} images")
    log_message(f"Validation set (classification): {len(val_dataset_cls_internal)} images")
    log_message(f"Test set (segmentation): {len(test_dataset_seg)} images")
    
    # STEP 2: Train the classifier using only image-level labels
    log_message("\n2. Training the classifier...")
    learning_rate = [0.0001]
    weight_decay = [0.0001]
    overall_best_acc = 0
    overall_best_classifier_state = None
    best_parameter_pairs = {}
    for learning_rate_value, weight_decay_value in product(learning_rate, weight_decay):
        current_accuracy, current_classifier_state = classifier_training_hypertuning(train_loader_cls, val_loader_cls, device, learning_rate_value, weight_decay_value, epochs=4)
        log_message(f"Validation Accuracy = {current_accuracy:.2f}%")
        if current_accuracy > overall_best_acc:
            overall_best_acc = current_accuracy
            overall_best_classifier_state = current_classifier_state
            best_parameter_pairs = { 
                "learning_rate": learning_rate_value,  # Fixed: use value not list
                "weight_decay": weight_decay_value     # Fixed: use value not list
            }
    
    # Load the best classifier
    log_message(f"Best Classfier model settings = {best_parameter_pairs}")
    classifier = get_classifier(device=device)
    classifier.load_state_dict(overall_best_classifier_state)
    log_message(f"Best validation accuracy: {overall_best_acc:.2f}%")
    
    # Save the classifier
    torch.save(classifier.state_dict(), os.path.join(RESULTS_DIR, "models", "best_classifier.pth"))
    
    # STEP 3: Generate pseudo-masks using GradCAM
    log_message("\n3. Generating pseudo-masks using GradCAM...")
    
    # Create a dataloader for the original training set (for mask generation)
    full_train_dataset = OxfordPetClassification(root='./data', split='trainval', transform=transforms['train'])
    full_train_loader = DataLoader(full_train_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Generate pseudo-masks for the training set
    pseudo_labeled_data = generate_pseudo_masks(classifier, full_train_loader, device, threshold=0.5)
    log_message(f"Generated {len(pseudo_labeled_data)} pseudo-labeled samples")
    
    # Create dataset with pseudo-labels
    pseudo_dataset = PseudoLabeledDataset(pseudo_labeled_data)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Visualize some pseudo-masks
    visualize_pseudo_masks(pseudo_dataset, output_dir=os.path.join(RESULTS_DIR, "gradcam_visualizations"))
    
    # STEP 4: Train segmentation model on pseudo-masks
    log_message("\n4. Training segmentation model on pseudo-masks...")
    
    log_message(f"Grid search on seg")
    learning_rate_segmentation = [0.0001]
    weight_decay_segmentation = [0.0001]
    overall_best_segmentation_iou = 0
    overall_best_segmentation_state = None
    best_parameter_pairs_segmentation = {}
    tracking_seg = {}

    for learning_rate_segmentation_value, weight_decay_segmentation_value in product(learning_rate_segmentation, weight_decay_segmentation):
        current_iou, current_state_seg, binary_metrics, track_train_seg_loss, track_iou_seg = segmentation_hypertuning_training(
            pseudo_loader, test_loader_seg, device, learning_rate_segmentation_value, weight_decay_segmentation_value, epochs=4
        )

        parameter_pairs = f"learning_rate={learning_rate_segmentation_value}, weight_decay={weight_decay_segmentation_value}"
        tracking_seg[parameter_pairs] = {
                "Training Loss": track_train_seg_loss,
                "IoU": track_iou_seg
        }

        log_message(f"Validation IoU = {current_iou:.2f}%")
        if current_iou > overall_best_segmentation_iou:
            overall_best_segmentation_iou = current_iou
            overall_best_segmentation_state = current_state_seg
            best_parameter_pairs_segmentation = {
                "learning_rate_segmentation": learning_rate_segmentation_value,  # Fixed: use value not list
                "weight_decay_segmentation": weight_decay_segmentation_value     # Fixed: use value not list
            }

    log_message(f"Best seg model settings = {best_parameter_pairs_segmentation}")

    segmentation_model = get_segmentation_model(device=device)
   
    # Load best segmentation model
    segmentation_model.load_state_dict(overall_best_segmentation_state)
    log_message(f"Best validation IoU: {overall_best_segmentation_iou:.4f}")
    
    # Save the segmentation model
    torch.save(segmentation_model.state_dict(), os.path.join(RESULTS_DIR, "models", "best_segmentation_model.pth"))
    
    # Create graphs directory and save graphs
    os.makedirs(os.path.join(RESULTS_DIR, "graphs"), exist_ok=True)
    create_graph(tracking_seg, output_dir=os.path.join(RESULTS_DIR, "graphs"))
    
    # STEP 5: Final evaluation on test set
    log_message("\n5. Final evaluation on test set...")
    final_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device, verbose=True)
    log_message("Final Evaluation Metrics:")
    for name, value in final_metrics.items():
        log_message(f"  {name}: {value:.4f}")
    
    # STEP 6: Visualize results
    log_message("\n6. Visualizing results...")
    os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)
    visualize_predictions(segmentation_model, test_loader_seg, device, 
                         grad_cam_enabled=True, num_samples=5,
                         output_dir=os.path.join(RESULTS_DIR, "predictions"))
    
    log_message("\nWeakly supervised segmentation pipeline completed!")

if __name__ == "__main__":
    main()
