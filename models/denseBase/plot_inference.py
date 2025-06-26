import _init_paths
import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dataload.dataloader import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, GNNPoseNet
from lib.transformations import quaternion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    cudnn.benchmark = True
    cudnn.deterministic = True
else:
    print('No GPU available, using CPU instead')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cudnn.benchmark = False
    cudnn.deterministic = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, required=True, help='dataset root dir')
parser.add_argument('--model', type=str, required=True, help='PoseNet model path')
parser.add_argument('--output_dir', type=str, default='plots/eval_linemod', help='Output directory for images')
parser.add_argument('--num_points', type=int, default=500, help='number of points to sample')
parser.add_argument('--img_size', type=int, default=480, help='image size for plotting')
parser.add_argument('--gnn', action='store_true', default=False, help='start training on the geometric model')
opt = parser.parse_args()

def main():
    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = opt.num_points
    bs = 1

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # Load models
    if opt.gnn:
        estimator = GNNPoseNet(num_points=num_points, num_obj=num_objects)
    else:
        estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.to(device)
    estimator.load_state_dict(torch.load(opt.model, map_location=device))
    estimator.eval()

    # Load dataset
    testdataset = PoseDataset_linemod(opt.dataset_root, 'eval', num_points=num_points, add_noise=False, noise_trans=0.0, device=device)

    # Select one image per object id
    selected_indices = {}
    for idx in [i*200 for i in range(len(testdataset)//200)]:
        # Skipping Yolo or Segnet Lost Detections
        while len(testdataset.__getitem__(idx)) == 1:
            idx += 1
        obj_id = testdataset.getObjectId(idx)
        if obj_id.item() not in selected_indices:
            print(f"Selected object {obj_id.item()} at index {idx}.")
            selected_indices[obj_id.item()] = idx
        if len(selected_indices) == len(objlist):
            break

    for obj_id, i in enumerate(objlist):
        if obj_id not in selected_indices:
            print(f"Object {i} not found in dataset.")
            continue
        idx = selected_indices[obj_id]
        data = testdataset[idx]
        points = data['cloud'].unsqueeze(0).to(device)
        choose = data['choose'].unsqueeze(0).to(device)
        img = data['image'].unsqueeze(0).to(device)
        if opt.gnn:
            graph_data = data['graph'].to(device)
        model_points = data['model_points'].to(device)
        obj_idx = data['obj_id'].to(device)
        if torch.is_tensor(obj_idx) and obj_idx.dim() == 0:
            obj_idx = obj_idx.item()
        obj_idx = torch.tensor([obj_idx], device=device)

        # Run PoseNet
        if opt.gnn:
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, graph_data, obj_idx, "color")
        else:
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, obj_idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        _, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

        # Transform model points with predicted pose
        model_points_np = model_points.cpu().detach().numpy()
        my_r_mat = quaternion_matrix(my_r)[:3, :3]
        pred_points = np.dot(model_points_np, my_r_mat.T) + my_t

        # Camera intrinsics
        cam_fx = 572.41140
        cam_fy = 573.57043
        cam_cx = 325.26110
        cam_cy = 242.04899

        # Project 3D points to 2D
        x = pred_points[:, 0]
        y = pred_points[:, 1]
        z = pred_points[:, 2]
        u = (x * cam_fx) / z + cam_cx
        v = (y * cam_fy) / z + cam_cy

        # Only keep points with positive depth
        valid = z > 0
        u = u[valid]
        v = v[valid]

        # Get RGB image for plotting
        rgb_img = data['rgb'].permute(1, 2, 0).cpu().numpy()

        # Plot and save
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(rgb_img)
        plt.scatter(u, v, s=1, c='r', label='Predicted Pose')
        plt.title(f'Object ID: {i}')
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        # Extract epochNumber and avgDis from estimator weight filename
        model_filename = os.path.basename(opt.model)
        # Expected format: pose_model_epochNumber_avgDis
        parts = model_filename.replace('.pth', '').split('_')
        if len(parts) >= 4:
            folder_name = f"{parts[2]}_{parts[3]}"
        else:
            folder_name = "unknown_model"
        save_dir = os.path.join(opt.output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'object_{i}_prediction.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization for object {i}.")

if __name__ == '__main__':
    main()