import os
from torch.utils.data import DataLoader
from dataloader import PoseDataset


if __name__ == '__main__':
    # CHECK FOR DATASET POSITION
    dataset_root = os.path.join(os.path.dirname(__file__), '../../dataset/linemod/DenseFusion/Linemod_preprocessed/')
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found at {dataset_root}. Please check the path.")
    print(f"Dataset found at {dataset_root}.")

    # DATASET TEST
    train_dataset = PoseDataset(
        dataset_root=dataset_root,
        split='train',
        train_ratio=0.8,
        seed=42
    )
    # DATASET PLOT TEST:
    idx = 0
    train_dataset.plotitem(idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


    print(f"Training samples: {len(train_dataset)}")