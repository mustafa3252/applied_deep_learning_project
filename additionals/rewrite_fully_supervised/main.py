# Gen AI Usage Statement:
# General architecture was designed and planned manually
# Gen AI was used to help with debugging, refactoring and refining the architecture in depth
# Gen AI used: Claude,Chatgpt and Co-Pilot

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
    get_binary_metrics,create_graph
)

def segmentation_hypertuning_training(pseudo_loader,test_loader_seg, device,learning_rate,weight_decay,epochs):

    track_train_seg_loss= []
    track_iou_seg = []
    segmentation_model = get_segmentation_model(device=device)
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
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
    return best_val_iou, best_seg_model_state, binary_metrics, track_train_seg_loss,track_iou_seg


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

    # Create data loaders for classification
    train_dataset_seg = OxfordPetSegmentation(root='./data', split='trainval',
                                             transform_img=transforms['val'],
                                             transform_mask=transforms['mask'])
    train_loader_seg = DataLoader(train_dataset_seg, batch_size=16, shuffle=False, num_workers=2)
    
    # Also load test set (only for final evaluation, with segmentation masks)
    test_dataset_seg = OxfordPetSegmentation(root='./data', split='test', 
                                          transform_img=transforms['val'],
                                          transform_mask=transforms['mask'])
    test_loader_seg = DataLoader(test_dataset_seg, batch_size=16, shuffle=False, num_workers=2)

    learning_rate_segmentation = [0.0001]
    weight_decay_segmentation = [0.0001]
    overall_best_segmentation_iou = 0
    overall_best_segmentation_state = None
    best_parameter_pairs_segmentation = {}
    tracking_seg = {}

    for learning_rate_segmentation_value, weight_decay_segmentation_value in product(learning_rate_segmentation, weight_decay_segmentation):

        currect_iou,current_state_seg,binary_metrics,track_train_seg_loss,track_iou_seg   = segmentation_hypertuning_training(
            train_loader_seg,test_loader_seg, device,learning_rate_segmentation_value,weight_decay_segmentation_value , epochs=1
        )

        parameter_pairs = f"learning_rate={learning_rate_segmentation_value}, weight_decay={weight_decay_segmentation_value}"
        tracking_seg[parameter_pairs] = {
                "Training Loss": track_train_seg_loss ,
                "IoU": track_iou_seg
        }

        log_message(f"Validation IoU = {currect_iou:.2f}%")
        if currect_iou > overall_best_segmentation_iou:
            overall_best_segmentation_iou = currect_iou
            overall_best_segmentation_state = current_state_seg
            best_parameter_pairs_segmentation = {"learning_rate_segmentation" : learning_rate_segmentation, "weight_decay_segmentation" : weight_decay_segmentation}


    log_message(f"Best seg model settings = {best_parameter_pairs_segmentation}")

    segmentation_model = get_segmentation_model(device=device)
   
    # Load best segmentation model
    segmentation_model.load_state_dict(overall_best_segmentation_state)
    log_message(f"Best validation IoU: {overall_best_segmentation_iou:.4f}")
    
    # Save the segmentation model
    torch.save(segmentation_model.state_dict(), "models/best_segmentation_model.pth")
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