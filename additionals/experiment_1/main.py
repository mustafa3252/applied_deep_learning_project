# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
 
# Gen AI used: Claude, ChatGPT and Co-Pilot

# For the changes in this specific experiment compared to the main code, Gen AI was not used

# Extra library Usage: We used more than three pip installs but this code was for the experiment and these extra libraries are not used for the final implementation of weakly supervised segmentation framework



import os
import json
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from itertools import product

# Import from local modules
from datasets import OxfordPetClassification, OxfordPetSegmentation, PseudoLabeledDataset
from models import get_classifier, get_segmentation_model, resnet_insert_dropout, fcn_insert_dropout
from utils import (
    set_seed, log_message, get_device, get_transforms, 
    GradCAM, generate_pseudo_masks, 
    train_classifier, train_segmentation, evaluate_segmentation,
    visualize_predictions, visualize_pseudo_masks,
    get_binary_metrics,create_graph
)


def classifier_training_hypertuning(classifier, train_loader_cls, val_loader_cls,
                                    device, learning_rate, weight_decay, epoch_range, data):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    num_epochs = epoch_range[-1] + 1  # Increase for better results
    best_val_acc = 0
    best_classifier_state = None
    
    for epoch in epoch_range:
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

        model_path = f"models/classifier_progress_{epoch}.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(classifier, model_path)

        data["cls"]["progress"] = {"epoch": epoch + 1, "model": model_path}
        save_progress(data)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_classifier_state = classifier.state_dict()

    data["cls"]["progress"] = None
    return best_val_acc, best_classifier_state


def segmentation_hypertuning_training(segmentation_model, pseudo_loader, test_loader_seg,
                                      device, learning_rate, weight_decay, epoch_range, data):
    track_train_seg_loss= []
    track_iou_seg = []
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    binary_metrics = get_binary_metrics(device)
    
    num_epochs = epoch_range[-1] + 1  # Increase for better results
    best_val_iou = 0
    best_seg_model_state = None
    
    for epoch in epoch_range:
        train_loss = train_segmentation(segmentation_model, pseudo_loader, optimizer, criterion, device)
        log_message(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
        track_train_seg_loss.append(train_loss)

        # Validate on test set
        eval_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device)
        track_iou_seg.append(eval_metrics["IoU"])
        log_message(f"Validation IoU: {eval_metrics['IoU']:.4f}, Dice: {eval_metrics['Dice']:.4f}")

        model_path = f"models/segmentation_model_progress_{epoch}.pth"
        os.makedirs("models", exist_ok=True)
        torch.save(segmentation_model, model_path)

        data["seg"]["progress"] = {"epoch": epoch + 1, "model": model_path}
        save_progress(data)

        # Save best model
        if eval_metrics['IoU'] > best_val_iou:
            best_val_iou = eval_metrics['IoU']
            best_seg_model_state = segmentation_model.state_dict()

    data["cls"]["progress"] = None
    return best_val_iou, best_seg_model_state, binary_metrics, track_train_seg_loss,track_iou_seg


def pairs_to_str(pairs: dict, pair_sep: str="_", kv_sep: str=""):
    return pair_sep.join([
        kv_sep.join([str(item) for item in pair])
        for pair in pairs.items()
    ])

def init_saved_progress():
    if os.path.exists(os.path.sep.join(["logs", "landmark.txt"])):
        with open("logs/landmark.txt", "r") as f:
            data = json.load(f)
    else:
        data = {"cls": {}, "seg": {}}
    
    return data

def save_progress(data):
    with open("logs/landmark.txt", "w") as f:
        json.dump(data, f)


def main():
    # Setup
    set_seed(42)
    device = get_device()
    transforms = get_transforms()
    
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    with open("logs/training_log.txt", "w") as f:
        f.write(f"=== Weakly Supervised Segmentation Pipeline - {datetime.now()} ===\n\n")
    
    log_message("Starting weakly supervised segmentation pipeline...")
    log_message(f"Device: {device}")
    log_message(f"PyTorch version: {torch.__version__}")
    
    data = init_saved_progress()

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
    learning_rate = [5e-05]
    weight_decay = [1e-05]
    resnet_probs = [0.1] #[0.1, 0.2, 0.25, 0.3]
    overall_best_acc = 0
    best_parameter_pairs = {}
    cls_num_epochs = 4

    for learning_rate_value, weight_decay_value, prob in product(learning_rate, weight_decay, resnet_probs):
        log_message(f"Dropout Probability Value: {prob}")

        parameter_pairs = {"learning_rate": learning_rate_value,
                           "weight_decay": weight_decay_value,
                           "resnet_prob": prob}
        parameter_pairs_str = pairs_to_str(parameter_pairs)

        model_data = data["cls"].get(parameter_pairs_str)
        if model_data is None:
            progress_data = data["cls"].get("progress")

            if progress_data is None or progress_data["epoch"] >= cls_num_epochs:
                model = resnet_insert_dropout(get_classifier(), prob, device=device)
                epoch_range = list(range(cls_num_epochs))

            else:
                model = torch.load(progress_data["model"], weights_only=False).to(device=device)
                epoch_range = list(range(progress_data["epoch"], cls_num_epochs))

            current_accuracy, current_classifier_state = classifier_training_hypertuning(
                model,
                train_loader_cls, val_loader_cls, device, learning_rate_value, weight_decay_value,
                epoch_range, data
            )
            log_message(f"Validation Accuracy = {current_accuracy:.2f}%")

            model_path = f"models/classifier_{parameter_pairs_str}.pth"
            save_model = resnet_insert_dropout(get_classifier(), prob, device=device)
            save_model.load_state_dict(current_classifier_state)
            os.makedirs("models", exist_ok=True)
            torch.save(save_model, model_path)

            data["cls"][parameter_pairs_str] = {"accuracy": current_accuracy, "model": model_path}
            save_progress(data)

        else:
            current_accuracy = model_data["accuracy"]
            log_message(f"Validation Accuracy = {current_accuracy:.2f}%")

        if current_accuracy > overall_best_acc:
            overall_best_acc = current_accuracy
            best_parameter_pairs = parameter_pairs

    # Load and save the best classifier
    log_message(f"Best Classfier model settings = {best_parameter_pairs}")
    classifier = torch.load(f"models/classifier_{pairs_to_str(best_parameter_pairs)}.pth", weights_only=False).to(device=device)
    torch.save(classifier, "models/best_classifier.pth")
    log_message(f"Best validation accuracy: {overall_best_acc:.2f}%")



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
    visualize_pseudo_masks(pseudo_dataset, output_dir="gradcam_visualizations")



    # STEP 4: Train segmentation model on pseudo-masks
    log_message("\n4. Training segmentation model on pseudo-masks...")
    
    log_message(f"Grid serach on seg")
    learning_rate_segmentation = [0.0001]
    weight_decay_segmentation = [0.0001]
    fcn_probs = [0.1, 0.2] #[0.1, 0.2, 0.3, 0.4]
    best_segmentation_iou = 0
    best_segmentation_parameter_pairs = {}
    tracking_seg = {}
    seg_num_epochs = 4

    for learning_rate_value, weight_decay_value, prob in product(learning_rate_segmentation, weight_decay_segmentation, fcn_probs):

        seg_param_pairs = {"learning_rate": learning_rate_value,
                                        "weight_decay": weight_decay_value,
                                        "fcn_prob": prob}
        seg_param_pairs_str = pairs_to_str(seg_param_pairs)

        model_data = data["seg"].get(seg_param_pairs_str)
        if model_data is None:
            progress_data = data["seg"].get("progress")

            if progress_data is None or progress_data["epoch"] >= seg_num_epochs:
                model = fcn_insert_dropout(get_segmentation_model(), prob, device=device)
                epoch_range = list(range(seg_num_epochs))

            else:
                model = torch.load(progress_data["model"], weights_only=False).to(device=device)
                epoch_range = list(range(progress_data["epoch"], seg_num_epochs))

            current_iou, current_state_seg, binary_metrics, track_train_seg_loss, track_iou_seg = segmentation_hypertuning_training(
                model,
                pseudo_loader, test_loader_seg, device, learning_rate_value, weight_decay_value,
                epoch_range, data
            )
            log_message(f"Validation IoU = {current_iou:.2f}%")

            model_path = f"models/segmentation_model_{seg_param_pairs_str}.pth"
            save_model = resnet_insert_dropout(get_segmentation_model(), prob, device=device)
            save_model.load_state_dict(current_state_seg)
            os.makedirs("models", exist_ok=True)
            torch.save(save_model, model_path)

            data["seg"][seg_param_pairs_str] = {
                "current_iou": current_iou,
                "track_train_seg_loss": track_train_seg_loss,
                "track_iou_seg": track_iou_seg,
                "model": model_path
            }
            save_progress(data)
        
        else:
            current_iou = model_data["current_iou"]
            track_train_seg_loss = model_data["track_train_seg_loss"]
            track_iou_seg = model_data["track_iou_seg"]

        parameter_pairs_key = pairs_to_str(seg_param_pairs, ", ", "=")
        tracking_seg[parameter_pairs_key] = {
            "Training Loss": track_train_seg_loss,
            "IoU": track_iou_seg,
        }

        if current_iou > best_segmentation_iou:
            best_segmentation_iou = current_iou
            best_segmentation_parameter_pairs = seg_param_pairs


    log_message(f"Best seg model settings = {best_segmentation_parameter_pairs}")
    log_message(f"Best validation IoU: {best_segmentation_iou:.4f}")

    # Load and save the best segmentation model
    segmentation_model = torch.load(f"models/segmentation_model_{pairs_to_str(best_segmentation_parameter_pairs)}.pth", weights_only=False).to(device=device)
    torch.save(segmentation_model, "models/best_segmentation_model.pth")
    create_graph(tracking_seg)



    # STEP 5: Final evaluation on test set
    log_message("\n5. Final evaluation on test set...")
    final_metrics = evaluate_segmentation(segmentation_model, test_loader_seg, binary_metrics, device, verbose=True)
    log_message("Final Evaluation Metrics:")
    for name, value in final_metrics.items():
        log_message(f"  {name}: {value:.4f}")
    
    # STEP 6: Visualize results
    log_message("\n6. Visualizing results...")
    visualize_predictions(segmentation_model, test_loader_seg, device, grad_cam_enabled=True, num_samples=5)
    
    log_message("\nWeakly supervised segmentation pipeline completed!")

if __name__ == "__main__":
    main()