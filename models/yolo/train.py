# --------------------------------------------------------
# 6D Object Pose Estimation - YOLO finetune
# Licensed under The MIT License [see LICENSE for details]
# Written by Fassio Simone
# --------------------------------------------------------
import os
import yaml
import torch
import ultralytics
from ultralytics import YOLO
import argparse

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check for MPS availability specifically
        if torch.backends.mps.is_available():
             device = torch.device("mps")
             print("Using MPS (Apple Silicon GPU)")
        else:
             # Fallback if MPS is listed but not truly available
             device = torch.device("cpu")
             print("MPS not available, using CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model on Linemod dataset.')
    parser.add_argument('--data', type=str, default='../../../dataset/yolo_linemod/linemod.yaml',
                        help='Path to the YOLO dataset config file (e.g., linemod.yaml).')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Pretrained YOLO model name or path (e.g., yolov8n.pt, yolov8x.pt).')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader.')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait for no improvement before early stopping.')
    parser.add_argument('--name', type=str, default='linemod_finetune', help='Name for the training run directory.')
    parser.add_argument('--save_dir', type=str, default='runs/detect', help='Directory to save training runs.')


    args = parser.parse_args()

    ultralytics.checks()

    # Set device
    device = get_device()
    # Use string format for device in YOLO functions
    device_str = str(device.index) if device.type == 'cuda' else device.type

    # --- Check for YOLO dataset config ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, args.data)
    if not os.path.exists(config_path):
        # Try alternative common location relative to script dir
        alt_config_path = os.path.join(script_dir, '../', args.data)
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
        else:
            raise FileNotFoundError(f"YOLO data config file not found at {config_path} or {alt_config_path}. Please run yolo_dataset_export.py first.")
    print(f"Using YOLO data config: {config_path}")
    # --- End Check ---

    # Load the YOLO model
    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    # Clear CUDA cache before training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
        # Check available GPU memory
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"GPU memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
        except Exception as e:
            print(f"Could not get GPU memory info: {e}")

    # Fine-tune model
    print("Starting fine-tuning...")
    results = model.train(
        data=config_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.save_dir, # Control the parent directory for runs
        device=device_str,
        patience=args.patience,
        save=True,
        pretrained=True, # Assumes loading a pretrained model like yolov8n.pt
        verbose=True,
        workers=args.workers,
        cache=True # Enable caching if RAM allows
    )

    # Path to best model
    # results.save_dir should be the full path to the specific run directory
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')

    if not os.path.exists(best_model_path):
        print(f"Warning: Best model not found at expected location: {best_model_path}")
        best_model_path = "N/A" # Indicate failure to find best model
    else:
        print(f"\nFine-tuning complete! Best model saved at: {best_model_path}")

        # Validate on validation set
        print("\nValidating fine-tuned model...")
        # Load the best model explicitly for validation
        best_model = YOLO(best_model_path)
        val_results = best_model.val(
            data=config_path,
            split='val', # Explicitly use validation split
            device=device_str
        )

        # Access metrics correctly from the Results object
        print(f"Validation mAP50-95: {val_results.box.map:.4f}")
        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Validation Precision: {val_results.box.mp:.4f}") # Mean Precision
        print(f"Validation Recall: {val_results.box.mr:.4f}") # Mean Recall

        # --- Optional: Save validation metrics ---
        val_metrics = {
            'mAP50-95': float(val_results.box.map),
            'mAP50': float(val_results.box.map50),
            'Precision': float(val_results.box.mp),
            'Recall': float(val_results.box.mr)
        }
        metrics_path = os.path.join(results.save_dir, 'validation_metrics.yaml')
        try:
            with open(metrics_path, 'w') as f:
                yaml.dump(val_metrics, f, default_flow_style=False)
            print(f"Validation metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Could not save validation metrics: {e}")
        # --- End Optional ---

if __name__ == '__main__':
    main()