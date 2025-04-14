import os
import torch
import numpy as np
from tqdm import tqdm
import yaml
import argparse 
from ultralytics import YOLO

from dataload.dataloader import PoseDataset

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def run_inference(dataset, model, num_samples=5, confidence_threshold=0.25, device=torch.device('cpu')):
    """Run YOLO inference on random samples from the dataset"""
    results_list = []
    
    # Select random indices
    indices = np.random.choice(len(dataset.samples), min(num_samples, len(dataset.samples)), replace=False)
    
    for i in indices:
        # Use the sample directly from dataset.samples to avoid the index issue
        folder_id, sample_id = dataset.samples[i]
        
        # LOADING PATHS - recreating the logic from __getitem__
        img_path = os.path.join(dataset.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        bbx_path = os.path.join(dataset.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
        
        # Load image
        img = dataset.load_image(img_path)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Move image to device if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.to(device)
        
        # Load bounding box safely using sample_id
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.safe_load(f)
            sample_data = bbx_data[sample_id][0]
            bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
            x, y, width, height = bbx[0], bbx[1], bbx[2], bbx[3]
        
        # Create a sample dict with just what we need
        sample = {
            'rgb': img,
            'top_left': (x, y),
            'bb_width': width,
            'bb_height': height,
            'folder_id': folder_id  # Add folder_id for class information
        }
        
        # Run inference with device specified
        results = model(img_np, conf=confidence_threshold, device=device.type)
        results_list.append((i, sample, results[0]))
    
    return results_list

def compute_metrics(dataset, model, iou_threshold=0.5, confidence_threshold=0.25, device=torch.device('cpu')):
    """Compute object detection metrics for the entire dataset"""
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(1, 16)}  # For each class
    
    total_samples = min(100, len(dataset.samples))  # Limit to 100 samples for speed
    
    for i in tqdm(range(total_samples), desc="Computing metrics"):
        # Use the sample directly to avoid the index issue
        folder_id, sample_id = dataset.samples[i]
        
        # LOADING PATHS
        img_path = os.path.join(dataset.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        bbx_path = os.path.join(dataset.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
        
        # Load image
        img = dataset.load_image(img_path)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Move image to device if it's a tensor
        if isinstance(img, torch.Tensor):
            img = img.to(device)
        
        # Load bounding box safely using sample_id
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.safe_load(f)
            sample_data = bbx_data[sample_id][0]
            bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
            x, y, width, height = bbx[0], bbx[1], bbx[2], bbx[3]
        
        # Get ground truth bounding box in [x1, y1, x2, y2] format
        gt_box = [x, y, x + width, y + height]
        gt_class = folder_id
        
        # Run inference with device specified
        results = model(img_np, conf=confidence_threshold, device=device.type)
        
        # Get predictions
        pred_boxes = []
        pred_classes = []
        if len(results[0].boxes) > 0:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()  # Get prediction boxes in xyxy format
            pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)  # Get predicted classes
        
        if len(pred_boxes) == 0:
            # No objects detected
            fn += 1
            class_metrics[gt_class]['fn'] += 1
            continue
        
        # Find the prediction with highest IoU
        best_iou = 0
        best_class = -1
        for idx, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_class = pred_classes[idx]
        
        if best_iou >= iou_threshold:
            tp += 1
            # Check if class is correct
            if best_class == gt_class - 1:  # Adjust for 0-indexing in YOLO
                class_metrics[gt_class]['tp'] += 1
            else:
                # Class is wrong but IoU is good
                class_metrics[gt_class]['fn'] += 1
                
            # Count extra detections as false positives
            fp += len(pred_boxes) - 1
        else:
            fp += len(pred_boxes)
            fn += 1
            class_metrics[gt_class]['fn'] += 1
    
    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-class metrics
    class_results = {}
    for class_id, metrics in class_metrics.items():
        class_tp = metrics['tp'] 
        class_fp = metrics['fp']
        class_fn = metrics['fn']
        
        class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0
        class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        class_results[f"class_{class_id:02d}"] = {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1_score": float(class_f1)
        }
    
    metrics = {
        "overall": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "per_class": class_results,
        "iou_threshold": float(iou_threshold),
        "confidence_threshold": float(confidence_threshold),
        "samples_evaluated": total_samples
    }
    
    return metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test fine-tuned YOLO model')
    parser.add_argument('--model', type=str, default='runs/detect/linemod_finetune/weights/best.pt',
                        help='Path to fine-tuned model')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    
    # Load the dataset
    dataset_root = os.path.join(os.path.dirname(__file__), '../dataset/linemod/DenseFusion/Linemod_preprocessed/')
    if not os.path.exists(dataset_root):
        dataset_root = os.path.join(os.path.dirname(__file__), 'dataset/linemod/DenseFusion/Linemod_preprocessed/')
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset not found at {dataset_root}. Please check the path.")
    
    val_dataset = PoseDataset(
        dataset_root=dataset_root,
        split='val',
        train_ratio=0.8,
        seed=42
    )
    
    print(f"Loaded validation dataset with {len(val_dataset)} samples.")
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    # Load fine-tuned model
    print(f"Loading fine-tuned YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directory
    save_dir = os.path.join(os.path.dirname(__file__), 'plots/yolo_finetuned')
    os.makedirs(save_dir, exist_ok=True)
    
    # Run inference on a few samples with device
    print("Running inference on random samples...")
    results = run_inference(val_dataset, model, num_samples=args.samples, device=device)
    
    # Plot and save results
    print("Plotting results...")
    plot_results(results, save_dir)
    
    # Compute metrics with device
    print("Computing metrics...")
    metrics = compute_metrics(val_dataset, model, device=device)
    
    # Print metrics
    print("\nObject Detection Metrics:")
    print(f"Overall precision: {metrics['overall']['precision']:.4f}")
    print(f"Overall recall: {metrics['overall']['recall']:.4f}")
    print(f"Overall F1 score: {metrics['overall']['f1_score']:.4f}")
    
    # Save metrics to YAML file
    with open(os.path.join(save_dir, "metrics.yaml"), 'w') as f:
        yaml.dump(metrics, f)
    
    print(f"Metrics saved to {save_dir}/metrics.yaml")

if __name__ == '__main__':
    main()