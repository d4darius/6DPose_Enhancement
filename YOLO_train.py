import os
import yaml
import shutil
import numpy as np
from tqdm import tqdm
import torch

# Import from local dataset
from dataload.dataloader import PoseDataset

def ensure_dependencies():
    """Ensure all dependencies are installed"""
    try:
        import ultralytics
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics")

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

def create_yolo_dataset(dataset, output_dir, split='train'):
    """Convert dataset to YOLO format"""
    # Create directories
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Creating YOLO dataset for {split} split...")
    
    for i in tqdm(range(len(dataset.samples))):
        folder_id, sample_id = dataset.samples[i]
        
        # Source paths
        img_path = os.path.join(dataset.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        bbx_path = os.path.join(dataset.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
        
        # Load image dimensions
        img = dataset.load_image(img_path)
        img_height, img_width = img.shape[1], img.shape[2]  # Height, Width
        
        # Load bounding box
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.safe_load(f)
            sample_data = bbx_data[sample_id][0]
            bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
            x, y, width, height = bbx[0], bbx[1], bbx[2], bbx[3]
        
        # Convert to YOLO format: class_id x_center y_center width height (normalized)
        class_id = folder_id - 1  # Convert to 0-indexed classes
        x_center = (x + width/2) / img_width
        y_center = (y + height/2) / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        # Ensure values are within bounds
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        # Copy image to YOLO dataset
        dest_img_path = os.path.join(images_dir, f"{folder_id:02d}_{sample_id:04d}.png")
        shutil.copy(img_path, dest_img_path)
        
        # Create YOLO label file
        label_path = os.path.join(labels_dir, f"{folder_id:02d}_{sample_id:04d}.txt")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
    
    return images_dir, labels_dir

def create_yolo_config(dataset_dir, num_classes=15):
    """Create YOLO configuration file"""
    config_path = os.path.join(dataset_dir, 'linemod.yaml')
    
    # Create class names list: 01-15
    class_names = [f"{i:02d}" for i in range(1, num_classes+1)]
    
    config = {
        'path': dataset_dir,
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def main():
    # Ensure dependencies
    ensure_dependencies()
    
    # Import YOLO after ensuring it's installed
    from ultralytics import YOLO
    
    # Set device
    device = get_device()
    
    # Set up paths
    dataset_root = os.path.join(os.path.dirname(__file__), '../dataset/linemod/DenseFusion/Linemod_preprocessed/')
    if not os.path.exists(dataset_root):
        dataset_root = os.path.join(os.path.dirname(__file__), 'dataset/linemod/DenseFusion/Linemod_preprocessed/')
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"Dataset not found at {dataset_root}. Please check the path.")
    
    yolo_dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset/yolo_linemod')
    os.makedirs(yolo_dataset_dir, exist_ok=True)
    
    # Create training dataset
    train_dataset = PoseDataset(
        dataset_root=dataset_root,
        split='train',
        train_ratio=0.1,
        seed=42
    )
    print(f"Loaded training dataset with {len(train_dataset)} samples.")
    train_imgs_dir, _ = create_yolo_dataset(train_dataset, yolo_dataset_dir, 'train')
    
    # Create validation dataset
    val_dataset = PoseDataset(
        dataset_root=dataset_root,
        split='val',
        train_ratio=0.1,
        seed=42
    )
    print(f"Loaded validation dataset with {len(val_dataset)} samples.")
    val_imgs_dir, _ = create_yolo_dataset(val_dataset, yolo_dataset_dir, 'val')
    
    # Create YOLO config file
    config_path = create_yolo_config(yolo_dataset_dir)
    print(f"Created YOLO config at {config_path}")
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Fine-tune model
    print("Starting fine-tuning...")
    results = model.train(
        data=config_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name='linemod_finetune',
        device=device.type,  # Use device.type instead of manual selection
        patience=10,  # Early stopping
        save=True,
        pretrained=True,
        verbose=True
    )
    
    # Path to best model
    best_model_path = os.path.join(os.path.dirname(__file__), 'runs/detect/linemod_finetune/weights/best.pt')
    
    # Validate on validation set
    print("\nValidating fine-tuned model...")
    val_results = YOLO(best_model_path).val(
        data=config_path,
        device=device.type  # Explicitly set device for validation
    )
    
    print("\nFine-tuning complete! Model saved at:", best_model_path)
    print(f"Validation metrics: {val_results}")
    
    # Export a report with metrics comparison
    print("\nCreating metrics comparison report...")
    
    # Load original metrics
    orig_metrics_path = os.path.join(os.path.dirname(__file__), 'plots/yolo_inference/metrics.yaml')
    if os.path.exists(orig_metrics_path):
        with open(orig_metrics_path, 'r') as f:
            orig_metrics = yaml.safe_load(f)
        
        # Create comparison report
        report = {
            'original_model': {
                'precision': orig_metrics.get('precision', 'N/A'),
                'recall': orig_metrics.get('recall', 'N/A'),
                'f1_score': orig_metrics.get('f1_score', 'N/A')
            },
            'finetuned_model': {
                'precision': val_results.box.map,  # mAP is similar to precision
                'recall': val_results.box.recall,
                'mAP50': val_results.box.map50,
                'mAP50-95': val_results.box.map
            }
        }
        
        # Save report
        report_path = os.path.join(os.path.dirname(__file__), 'plots/yolo_finetune/metrics_comparison.yaml')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            yaml.dump(report, f)
        
        print(f"Metrics comparison saved to {report_path}")

if __name__ == '__main__':
    main()