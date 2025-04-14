import os
import yaml
import shutil
import numpy as np
from tqdm import tqdm
import torch
import multiprocessing  # Import multiprocessing
from functools import partial  # Import partial for worker function arguments
from collections import defaultdict  # To group samples by folder
import ultralytics
from ultralytics import YOLO

# Import from local dataset
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

def process_folder(folder_args, dataset_root, output_dir, split):
    """Worker function to process all samples within a single folder (class) for YOLO format conversion."""
    folder_id, sample_ids_for_folder = folder_args
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')

    bbx_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
    try:
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load or parse gt.yml for folder {folder_id}, skipping folder. Error: {e}")
        return
      
    img_width, img_height = None, None
    if not sample_ids_for_folder:
        print(f"Warning: No samples provided for folder {folder_id} in split '{split}'. Skipping folder.")
        return

    first_sample_id = sample_ids_for_folder[0]
    first_img_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", f"rgb/{first_sample_id:04d}.png")
    try:
        from PIL import Image
        with Image.open(first_img_path) as img_pil:
            img_width, img_height = img_pil.size  # Width, Height
    except Exception as e:
        print(f"Warning: Could not read image dimensions from first image {first_img_path} for folder {folder_id}, skipping folder. Error: {e}")
        return

    # Process each sample within this folder
    for sample_id in sample_ids_for_folder:
        # Source image path
        img_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")

        # Basic check if source image exists (optional, as first image check might suffice)
        if not os.path.exists(img_path):
            print(f"Warning: Source image not found, skipping sample {sample_id} in folder {folder_id}: {img_path}")
            continue

        # Load bounding box from pre-loaded data
        try:
            sample_data = bbx_data[sample_id][0]
            bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
            x, y, width, height = bbx[0], bbx[1], bbx[2], bbx[3]
        except (KeyError, TypeError, IndexError) as e:
            print(f"Warning: Skipping sample {sample_id} in folder {folder_id} due to missing/invalid data in pre-loaded gt.yml. Error: {e}")
            continue

        # Convert to YOLO format using cached dimensions
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
        try:
            if not os.path.exists(dest_img_path):  # Avoid re-copying if possible
                 shutil.copy(img_path, dest_img_path)
        except Exception as e:
            print(f"Warning: Could not copy image {img_path} to {dest_img_path}. Error: {e}")
            continue # Skip label creation if image copy failed

        # Create YOLO label file
        label_path = os.path.join(labels_dir, f"{folder_id:02d}_{sample_id:04d}.txt")
        try:
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")
        except Exception as e:
             print(f"Warning: Could not write label file {label_path}. Error: {e}")

def create_yolo_dataset_parallel(dataset, output_dir, split='train', num_workers=None):
    """Convert dataset to YOLO format in parallel, processing folder by folder."""
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Check if dataset seems complete based on expected file count
    num_expected_files = len(dataset.samples)
    # Add a small tolerance in case some files were skipped due to errors
    if len(os.listdir(images_dir)) >= num_expected_files * 0.95 and len(os.listdir(labels_dir)) >= num_expected_files * 0.95:
         print(f"YOLO dataset for '{split}' split already exists and seems reasonably complete. Skipping conversion.")
         return images_dir, labels_dir

    print(f"Creating YOLO dataset for '{split}' split using parallel processing (folder-wise)...")

    # --- Group samples by folder_id ---
    samples_by_folder = defaultdict(list)
    for folder_id, sample_id in dataset.samples:
        samples_by_folder[folder_id].append(sample_id)
    folder_args_list = list(samples_by_folder.items()) # List of (folder_id, [sample_id1, sample_id2,...])
    # --- End Grouping ---

    if num_workers is None:
        num_workers = min(os.cpu_count(), len(folder_args_list)) # Use available cores, but not more than #folders
        print(f"Using {num_workers} workers for dataset conversion.")

    # Create a partial function with fixed arguments for the worker
    worker_func = partial(process_folder, dataset_root=dataset.dataset_root, output_dir=output_dir, split=split)

    # Use multiprocessing Pool to process folders in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, folder_args_list), total=len(folder_args_list), desc=f"Converting {split} folders"))

    print(f"Finished creating YOLO dataset for '{split}' split.")
    return images_dir, labels_dir

def create_yolo_config(dataset_dir, num_classes=15):
    """Create YOLO configuration file"""
    config_path = os.path.join(dataset_dir, 'linemod.yaml')

    # Create class names list: 01-15
    class_names = [f"{i:02d}" for i in range(1, num_classes+1)]

    config = {
        'path': os.path.abspath(dataset_dir), # Use absolute path
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path

def main():
    ultralytics.checks()

    # Set device
    device = get_device()

    # Set up paths
    # Try relative path first, then absolute if needed
    relative_dataset_root = '../dataset/linemod/DenseFusion/Linemod_preprocessed/'
    absolute_dataset_root = '/Users/simone/Documents/GitHub/kokkuapples/dataset/linemod/DenseFusion/Linemod_preprocessed/' # Adjust if needed

    if os.path.exists(os.path.join(os.path.dirname(__file__), relative_dataset_root)):
         dataset_root = os.path.join(os.path.dirname(__file__), relative_dataset_root)
    elif os.path.exists(absolute_dataset_root):
         dataset_root = absolute_dataset_root
    else:
         raise FileNotFoundError(f"Dataset not found at relative path {relative_dataset_root} or absolute path {absolute_dataset_root}. Please check the path.")
    print(f"Using dataset root: {dataset_root}")

    yolo_dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset/yolo_linemod')
    os.makedirs(yolo_dataset_dir, exist_ok=True)

    # --- Dataset Creation (Optimized) ---
    # Create training dataset object (needed for sample list)
    train_dataset_obj = PoseDataset(
        dataset_root=dataset_root,
        split='train',
        train_ratio=0.8, # Using 80% for training
        seed=42
    )
    print(f"Loaded training dataset definition with {len(train_dataset_obj.samples)} samples.")
    # Create YOLO format dataset in parallel (folder-wise, or skip if exists)
    train_imgs_dir, _ = create_yolo_dataset_parallel(train_dataset_obj, yolo_dataset_dir, 'train')

    # Create validation dataset object
    val_dataset_obj = PoseDataset(
        dataset_root=dataset_root,
        split='val',
        train_ratio=0.8, # Using the remaining 20% for validation
        seed=42
    )
    print(f"Loaded validation dataset definition with {len(val_dataset_obj.samples)} samples.")
    # Create YOLO format dataset in parallel (folder-wise, or skip if exists)
    val_imgs_dir, _ = create_yolo_dataset_parallel(val_dataset_obj, yolo_dataset_dir, 'val')
    # --- End Dataset Creation ---

    # Create YOLO config file
    config_path = create_yolo_config(yolo_dataset_dir)
    print(f"Created YOLO config at {config_path}")

    # Load the YOLOv11 model
    model_name = 'yolo11n.pt' # Largest official model
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
        # Check available GPU memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")

    # Fine-tune model
    print("Starting fine-tuning...")
    results = model.train(
        data=config_path,
        epochs=50,
        imgsz=640,
        batch=32,
        name='linemod_finetune_yolo11n', # Updated name
        device=device.type,
        patience=10,
        save=True,
        pretrained=True,
        verbose=True,
        workers=8
    )

    # Path to best model (adjust based on the 'name' parameter)
    best_model_path = os.path.join(os.path.dirname(__file__), f'runs/detect/{results.save_dir.split("/")[-1]}/weights/best.pt') # Dynamically get path

    # Validate on validation set
    print("\nValidating fine-tuned model...")
    # Load the best model explicitly for validation
    best_model = YOLO(best_model_path)
    val_results = best_model.val(
        data=config_path,
        device=device.type
    )

    print("\nFine-tuning complete! Model saved at:", best_model_path)
    # Access metrics correctly from the Results object
    print(f"Validation mAP50-95: {val_results.box.map}")
    print(f"Validation mAP50: {val_results.box.map50}")
    print(f"Validation Precision: {val_results.box.mp}") # Mean Precision
    print(f"Validation Recall: {val_results.box.mr}") # Mean Recall

    # Export a report with metrics comparison
    print("\nCreating metrics comparison report...")

    # Load original metrics if they exist
    orig_metrics = {}
    orig_metrics_path = os.path.join(os.path.dirname(__file__), 'plots/yolo_inference/metrics.yaml')
    if os.path.exists(orig_metrics_path):
        try:
            with open(orig_metrics_path, 'r') as f:
                orig_metrics_data = yaml.safe_load(f)
                # Extract overall metrics if available
                if 'overall' in orig_metrics_data:
                    orig_metrics = orig_metrics_data['overall']
                else: # Fallback for older format
                    orig_metrics = orig_metrics_data

        except Exception as e:
            print(f"Could not load or parse original metrics file: {e}")

    # Create comparison report
    report = {
        'original_model (yolo11n)': {
            'precision': orig_metrics.get('precision', 'N/A'),
            'recall': orig_metrics.get('recall', 'N/A'),
            'f1_score': orig_metrics.get('f1_score', 'N/A')
        },
        'finetuned_model (yolo11n)': {
            'precision': float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 'N/A',
            'recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 'N/A',
            'mAP50': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 'N/A',
            'mAP50-95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 'N/A'
        }
    }

    # Save report
    report_dir = os.path.join(os.path.dirname(__file__), 'plots/yolo_finetune_yolo11n') # Updated directory name
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'metrics_comparison.yaml')
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)

    print(f"Metrics comparison saved to {report_path}")

if __name__ == '__main__':
    # Add this guard for multiprocessing on Windows/macOS
    multiprocessing.freeze_support()
    main()