import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import open3d as o3d
from PIL import Image

class PoseDataset(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):

        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.samples = []
        self.load_dataset()

    def load_dataset(self):
        f = []
        names = [int(n.split("/")[-1]) for n in sorted([f for f in os.listdir(os.path.join(self.dataset_root, 'data')) if os.path.isdir(os.path.join(self.dataset_root, 'data', f))])]

        for folder in names:
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder:02d}")
            samples = len(os.listdir(os.path.join(folder_path, 'rgb')))
            f.append((folder, samples))
        all_samples = [(folder, sample_id) for folder, s in f for sample_id in range(s)]
        if self.split == 'train':
            self.samples = train_test_split(all_samples, train_size=self.train_ratio, random_state=self.seed)[0]
        elif self.split == 'val':
            self.samples = train_test_split(all_samples, train_size=self.train_ratio, random_state=self.seed)[1]

    #Define here some usefull functions to access the data
    def load_image(self, img_path):
        # Load an RGB image and convert to tensor.
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
    def load_depth(self, depth_path):
        # Load a depth image and convert to tensor.
        depth = Image.open(depth_path).convert("L")
        return self.transform(depth)
    def load_mask(self, mask_path):
        # Load a mask image and convert to tensor.
        mask = Image.open(mask_path).convert("RGBA")
        return self.transform(mask)
    def load_model(self, model_path):
        # Load a 3D model.
        mesh = o3d.io.read_triangle_mesh(model_path)
        mesh.compute_vertex_normals()
        return mesh
    def load_pose(self, pose_path, idx):
        # Load a 6D pose.
        with open(pose_path, 'r') as f:
            pose_data = yaml.safe_load(f)
            sample_data = pose_data[idx][0]
            # Convert to numpy array
            rot_mat = np.array(sample_data['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            tras_vec = np.array(sample_data['cam_t_m2c'], dtype=np.float32)
        return rot_mat, tras_vec
    def load_bbx(self, bbx_path, idx):
        # Load a bounding box.
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.safe_load(f)
            sample_data = bbx_data[idx][0]
            # Convert to numpy array
            bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
        return bbx[0], bbx[1], bbx[2], bbx[3]

    def __len__(self):
        #Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        # LOADING PATHS
        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"depth/{sample_id:04d}.png")
        mask_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"mask/{sample_id:04d}.png")
        model_path = os.path.join(self.dataset_root, 'models', f"obj_{folder_id:02d}.ply")
        bbx_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
        pose_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")

        # DATA OUTPUT
        img = self.load_image(img_path)
        depth = self.load_depth(depth_path)
        mask = self.load_mask(mask_path)
        model = self.load_model(model_path)
        rot_mat, tras_vec = self.load_pose(pose_path, idx)
        x, y, width, height = self.load_bbx(bbx_path, idx)

        return {
            "rgb": img,
            "depth": depth,
            "mask": mask,
            "model": model,
            "top_left": (x, y),
            "bb_width": width,
            "bb_height": height,
            "rotation_matrix": rot_mat,
            "translation_vector": tras_vec
        }
    
    def plotitem(self, idx, show=True):
        img, depth, mask, model, (x, y), width, height, rot_mat, t_vec = self.__getitem__(idx).values()
        fix, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].imshow(img.permute(1, 2, 0))
        ax[0].set_title("RGB Image")
        # Adding the bounding box
        edges = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]])
        # Create a rectangle patch
        plt.Polygon(edges, closed=True, fill=None, edgecolor='r', linewidth=2)
        ax[0].add_patch(plt.Polygon(edges, closed=True, fill=None, edgecolor='r', linewidth=2))
        ax[1].imshow(depth.permute(1, 2, 0), cmap='gray')
        ax[1].set_title("Depth Image")
        ax[2].imshow(mask.permute(1, 2, 0))
        ax[2].set_title("Mask Image")
        # Convert Open3D mesh to numpy array for visualization
        mesh = np.asarray(model.vertices)
        # Create a 3D scatter plot
        if show:
            # We plot the 3D visualization with the rotation matrix and translation vector applied
            mesh = np.dot(mesh, rot_mat.T) + t_vec
        fig = go.Figure(data=[go.Scatter3d(
            x=mesh[:, 0],
            y=mesh[:, 1],
            z=mesh[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue')
        )])
        fig.update_layout(scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ))
        # Create the directory if it doesn't exist
        if not(os.path.exists(os.path.join(os.path.dirname(__file__), '../plots/testing'))):
            os.makedirs(os.path.join(os.path.dirname(__file__), '../plots/testing'))
        # Remove the previous plots if they exist
        if os.path.exists(os.path.join(os.path.dirname(__file__), '../plots/testing', f"Images_plot_{idx}.png")):
            os.remove(os.path.join(os.path.dirname(__file__), '../plots/testing', f"Images_plot_{idx}.png"))
        if os.path.exists(os.path.join(os.path.dirname(__file__), '../plots/testing', f"3D_plot_{idx}.html")):
            os.remove(os.path.join(os.path.dirname(__file__), '../plots/testing', f"3D_plot_{idx}.html"))
        # Save the plots
        fix.savefig(os.path.join(os.path.dirname(__file__), '../plots/testing', f"Images_plot_{idx}.png"))
        fig.write_image(os.path.join(os.path.dirname(__file__), '../plots/testing', f"3D_plot_{idx}.png"))
        if show:
            plt.show()
        else:
            plt.close()