import os
import yaml
import shutil
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
from collections import defaultdict
from PIL import Image
import argparse
from dataload.dataloader import PoseDataset

def process_folder(folder_args, dataset_root, output_dir, split):
    """Worker function to process all samples within a single folder (class) for YOLO format conversion."""
    folder_id, sample_ids_for_folder = folder_args
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')

    # --- Optimization: Load GT file once per folder ---
    bbx_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
    try:
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Warning: Could not load or parse gt.yml for folder {folder_id}, skipping folder. Error: {e}")
        return
    # --- End Optimization ---

    # --- Optimization: Load image dimensions once per folder (using the first sample) ---
    img_width, img_height = None, None
    if not sample_ids_for_folder:
        print(f"Warning: No samples provided for folder {folder_id} in split '{split}'. Skipping folder.")
        return

    first_sample_id = sample_ids_for_folder[0]
    first_img_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", f"rgb/{first_sample_id:04d}.png")
    try:
        with Image.open(first_img_path) as img_pil:
            img_width, img_height = img_pil.size  # Width, Height
    except Exception as e:
        print(f"Warning: Could not read image dimensions from first image {first_img_path} for folder {folder_id}, skipping folder. Error: {e}")
        return
    # --- End Optimization ---

    # Process each sample within this folder
    for sample_id in sample_ids_for_folder:
        # Source image path
        img_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")

        # Basic check if source image exists
        if not os.path.exists(img_path):
            print(f"Warning: Source image not found, skipping sample {sample_id} in folder {folder_id}: {img_path}")
            continue

        # Load bounding box from pre-loaded data
        try:
            sample_data = bbx_data[sample_id]
            # Select the bbx of the corresponding obj id
            for sample in sample_data:
                if sample['obj_id'] == folder_id:
                    sample_data = sample
                    break
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
    elif num_workers == 0: # Allow user to specify 0 for sequential processing
        num_workers = 1
        print("Using 1 worker (sequential processing) for dataset conversion.")


    # Create a partial function with fixed arguments for the worker
    worker_func = partial(process_folder, dataset_root=dataset.dataset_root, output_dir=output_dir, split=split)

    # Use multiprocessing Pool to process folders in parallel (or run sequentially if num_workers=1)
    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(worker_func, folder_args_list), total=len(folder_args_list), desc=f"Converting {split} folders"))
    else:
        for args in tqdm(folder_args_list, desc=f"Converting {split} folders sequentially"):
            worker_func(args)


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

    print(f"Created YOLO config at {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Convert Linemod dataset to YOLO format.')
    parser.add_argument('--dataset_root', type=str, default='../../dataset/linemod/DenseFusion/Linemod_preprocessed/',
                        help='Path to the root of the preprocessed Linemod dataset.')
    parser.add_argument('--output_dir', type=str, default='../../dataset/yolo_linemod',
                        help='Directory to save the YOLO formatted dataset.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val split.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for parallel processing (default: all cores, 0 for sequential).')

    args = parser.parse_args()

    # Adjust relative paths to be relative to the script's location (dataload folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root_abs = os.path.abspath(os.path.join(script_dir, args.dataset_root))
    output_dir_abs = os.path.abspath(os.path.join(script_dir, args.output_dir))

    if not os.path.exists(dataset_root_abs):
         raise FileNotFoundError(f"Original dataset not found at {dataset_root_abs}. Please check the path.")
    print(f"Using original dataset root: {dataset_root_abs}")

    os.makedirs(output_dir_abs, exist_ok=True)
    print(f"Outputting YOLO dataset to: {output_dir_abs}")

    # --- Dataset Creation ---
    # Create training dataset object (needed for sample list)
    train_dataset_obj = PoseDataset(
        dataset_root=dataset_root_abs,
        split='train',
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    print(f"Loaded training dataset definition with {len(train_dataset_obj.samples)} samples.")
    # Create YOLO format dataset in parallel (folder-wise, or skip if exists)
    create_yolo_dataset_parallel(train_dataset_obj, output_dir_abs, 'train', args.num_workers)

    # Create validation dataset object
    val_dataset_obj = PoseDataset(
        dataset_root=dataset_root_abs,
        split='val',
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    print(f"Loaded validation dataset definition with {len(val_dataset_obj.samples)} samples.")
    # Create YOLO format dataset in parallel (folder-wise, or skip if exists)
    create_yolo_dataset_parallel(val_dataset_obj, output_dir_abs, 'val', args.num_workers)
    # --- End Dataset Creation ---

    # Create YOLO config file
    create_yolo_config(output_dir_abs)

    print("\nYOLO dataset export complete.")


if __name__ == '__main__':
    # Add this guard for multiprocessing compatibility
    multiprocessing.freeze_support()
    main()