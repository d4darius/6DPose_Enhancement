import os
import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import glob

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

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

def load_yolo_ground_truth(label_path, img_width, img_height):
    """Loads ground truth boxes from a YOLO label file."""
    gt_boxes = []
    gt_classes = []
    if not os.path.exists(label_path):
        return gt_boxes, gt_classes # Return empty lists if no label file

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])

                # Convert YOLO format (normalized center x, center y, width, height)
                # to pixel coordinates [x1, y1, x2, y2]
                box_width = width_norm * img_width
                box_height = height_norm * img_height
                x_center = x_center_norm * img_width
                y_center = y_center_norm * img_height

                x1 = x_center - box_width / 2
                y1 = y_center - box_height / 2
                x2 = x_center + box_width / 2
                y2 = y_center + box_height / 2

                gt_boxes.append([x1, y1, x2, y2])
                gt_classes.append(class_id)
    return gt_boxes, gt_classes


def compute_manual_metrics_yolo_format(yolo_val_img_dir, model, iou_threshold, confidence_threshold, device):
    """Compute manual P/R/F1 metrics using YOLO formatted dataset."""
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    # Assuming classes 0-14 indices based on YOLO format
    class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(15)} # Adjust range if num_classes differs
    total_samples_processed = 0

    image_files = sorted(glob.glob(os.path.join(yolo_val_img_dir, '*.png'))) # Or other extensions like .jpg
    if not image_files:
        print(f"Error: No image files found in {yolo_val_img_dir}")
        return None

    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        for img_path in tqdm(image_files, desc="Computing Manual Metrics"):
            total_samples_processed += 1
            label_path = img_path.replace('/images/', '/labels/').replace('.png', '.txt') # Construct label path

            # Load Image
            try:
                img = Image.open(img_path).convert("RGB")
                img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
                continue

            # Load Ground Truth
            gt_boxes, gt_classes_indices = load_yolo_ground_truth(label_path, img_width, img_height)

            # Perform inference
            results = model(img, conf=confidence_threshold, device=device, verbose=False)[0] # Get results for the single image

            # Get predictions
            pred_boxes = []
            pred_classes_indices = []
            if len(results.boxes) > 0:
                pred_boxes = results.boxes.xyxy.cpu().numpy()
                pred_classes_indices = results.boxes.cls.cpu().numpy().astype(int)

            # --- Matching Logic (Handling multiple GTs and Preds) ---
            # Based on standard detection metrics calculation
            # Create lists to track matched GTs and Preds
            gt_matched = [False] * len(gt_boxes)
            pred_matched = [False] * len(pred_boxes)

            # Iterate through predictions, sorted by confidence (optional but common)
            # If confidence scores are available: sorted_indices = np.argsort(-results.boxes.conf.cpu().numpy())
            # else: iterate 0 to N-1
            for pred_idx in range(len(pred_boxes)):
                pred_box = pred_boxes[pred_idx]
                pred_class_idx = pred_classes_indices[pred_idx]

                best_iou = 0
                best_gt_idx = -1

                # Find the best matching GT box (above threshold, same class, not already matched)
                for gt_idx in range(len(gt_boxes)):
                    if gt_matched[gt_idx]: continue # Skip already matched GT
                    if gt_classes_indices[gt_idx] != pred_class_idx: continue # Skip if class doesn't match

                    iou = calculate_iou(pred_box, gt_boxes[gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # If a match is found above the threshold
                if best_iou >= iou_threshold:
                    # Mark both GT and Pred as matched
                    gt_matched[best_gt_idx] = True
                    pred_matched[pred_idx] = True
                    # Increment True Positive count for the class
                    tp += 1
                    if pred_class_idx in class_metrics: class_metrics[pred_class_idx]['tp'] += 1
                # else: This prediction is a False Positive (handled below)

            # Count False Positives: Predictions that were not matched to any GT
            for pred_idx in range(len(pred_boxes)):
                if not pred_matched[pred_idx]:
                    fp += 1
                    pred_class_idx = pred_classes_indices[pred_idx]
                    if pred_class_idx in class_metrics: class_metrics[pred_class_idx]['fp'] += 1

            # Count False Negatives: Ground Truths that were not matched by any prediction
            for gt_idx in range(len(gt_boxes)):
                if not gt_matched[gt_idx]:
                    fn += 1
                    gt_class_idx = gt_classes_indices[gt_idx]
                    if gt_class_idx in class_metrics: class_metrics[gt_class_idx]['fn'] += 1
            # --- End Matching Logic ---


    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate per-class metrics
    class_results = {}
    for class_id, metrics_dict in class_metrics.items():
        class_tp = metrics_dict['tp']
        class_fp = metrics_dict['fp']
        class_fn = metrics_dict['fn']

        class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0
        class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0

        # Use 0-based index for key, map to name later if needed
        class_results[class_id] = {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1_score": float(class_f1),
            "tp": class_tp,
            "fp": class_fp,
            "fn": class_fn
        }

    manual_metrics = {
        "overall": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "per_class": class_results, # Keys are 0-14
        "iou_threshold": float(iou_threshold),
        "confidence_threshold": float(confidence_threshold),
        "samples_evaluated": total_samples_processed
    }

    return manual_metrics


def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned YOLO model on Linemod validation set.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the best.pt model weights.')
    parser.add_argument('--data', type=str, default="../../../dataset/yolo_linemod/linemod.yaml",
                        help='Path to the YOLO dataset config file (e.g., linemod.yaml).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for model.val() (mAP calculation).')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for model.val() (mAP calculation).')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for mAP and manual metrics.')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for predictions.')
    parser.add_argument('--save_dir', type=str, default='test_results', help='Directory to save test results (e.g., metrics.yaml).')


    args = parser.parse_args()

    # --- Setup ---
    device = get_device()
    device_str = str(device.index) if device.type == 'cuda' else device.type
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model from: {args.weights}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found at {args.weights}")
    model = YOLO(args.weights)
    model.to(device)

    # --- Load YOLO Config and Paths ---
    print(f"Loading YOLO data config from: {args.data}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"YOLO data config not found at {args.data}")
    with open(args.data, 'r') as f:
        yolo_config = yaml.safe_load(f)
    class_names = yolo_config.get('names', {}) # Get class names {0: '01', 1: '02', ...}
    yolo_dataset_path = yolo_config.get('path', os.path.dirname(args.data)) # Get base path from config or assume it's relative to config
    yolo_val_img_dir = os.path.join(yolo_dataset_path, yolo_config.get('val', 'val/images')) # Get val images path

    if not os.path.isdir(yolo_val_img_dir):
        raise NotADirectoryError(f"YOLO validation image directory not found: {yolo_val_img_dir}")
    print(f"Using YOLO validation images from: {yolo_val_img_dir}")


    # --- Calculate mAP using model.val() ---
    print("\nCalculating mAP using model.val()...")
    val_results = model.val(
        data=args.data,
        split='val',
        device=device_str,
        iou=args.iou_threshold,
        conf=args.conf_threshold,
        batch=args.batch_size,
        workers=args.workers,
        verbose=True
    )
    map_metrics = {
        "mAP50-95": float(val_results.box.map) if hasattr(val_results.box, 'map') else 'N/A',
        "mAP50": float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 'N/A',
        "Precision(val)": float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 'N/A',
        "Recall(val)": float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 'N/A'
    }
    print("mAP Calculation Complete.")
    print(f"  mAP@0.5:0.95 = {map_metrics['mAP50-95']}")
    print(f"  mAP@0.5 = {map_metrics['mAP50']}")

    # --- Compute Manual Metrics (P/R/F1) using YOLO format ---
    print("\nComputing manual metrics using YOLO format dataset...")
    manual_metrics = compute_manual_metrics_yolo_format(
        yolo_val_img_dir, model, args.iou_threshold, args.conf_threshold, device
    )

    if manual_metrics is None:
        print("Manual metrics computation failed.")
        return # Exit if computation failed

    # Map class indices in manual metrics to names
    manual_metrics["per_class_named"] = {
        class_names.get(idx, f"class_{idx}"): metrics
        for idx, metrics in manual_metrics["per_class"].items()
    }


    # --- Print & Save All Metrics ---
    print("\n--- Combined Metrics Summary ---")
    print("[mAP Metrics (calculated by YOLO)]")
    for key, value in map_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n[Manual Metrics (P/R/F1 @ IoU={:.2f}, Conf={:.2f})]".format(args.iou_threshold, args.conf_threshold))
    print(f"  Overall Precision: {manual_metrics['overall']['precision']:.4f}")
    print(f"  Overall Recall:    {manual_metrics['overall']['recall']:.4f}")
    print(f"  Overall F1 Score:  {manual_metrics['overall']['f1_score']:.4f}")
    print(f"  Samples Evaluated: {manual_metrics['samples_evaluated']}")

    print("\n  Per-Class Manual Metrics:")
    # Ensure class_names mapping worked, otherwise use index
    if manual_metrics.get("per_class_named"):
        for name, metrics in manual_metrics["per_class_named"].items():
             print(f"    Class {name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f} (TP:{metrics['tp']}, FP:{metrics['fp']}, FN:{metrics['fn']})")
    else:
         for idx, metrics in manual_metrics["per_class"].items():
             print(f"    Class {idx}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f} (TP:{metrics['tp']}, FP:{metrics['fp']}, FN:{metrics['fn']})")


    # Combine metrics for saving
    all_metrics = {
        "map_metrics": map_metrics,
        "manual_metrics": manual_metrics # Includes overall, per_class (index), per_class_named
    }

    # Save metrics to YAML file
    metrics_path = os.path.join(args.save_dir, "test_metrics.yaml")
    try:
        with open(metrics_path, 'w') as f:
            yaml.dump(all_metrics, f, default_flow_style=False, sort_keys=False, width=1000)
        print(f"\nAll metrics saved to {metrics_path}")
    except Exception as e:
        print(f"\nError saving metrics: {e}")


if __name__ == '__main__':
    main()