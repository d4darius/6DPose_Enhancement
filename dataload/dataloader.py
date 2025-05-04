import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy.ma as ma
import torch
from ultralytics import YOLO

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

class PoseDataset(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42, num_points=500, add_noise=False, noise_trans=0.03, refine=False):

        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.num_points = num_points
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.refine = refine
        self.args = os.path.join(os.path.dirname(__file__), '../models/yolo/weights/best.pt')
        self.model = YOLO(self.args)

        # Object list and metadata
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.pt = {}

        # Camera intrinsics
        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.img_width = 480
        self.img_length = 640
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.samples = []
        self.load_dataset()
        self.load_metadata()

    def load_dataset(self):
        names = [int(n.split("/")[-1]) for n in sorted([f for f in os.listdir(os.path.join(self.dataset_root, 'data')) if os.path.isdir(os.path.join(self.dataset_root, 'data', f))])]

        for folder in names:
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder:02d}")
            if self.split == 'train':
                sample_path = os.path.join(folder_path, 'train.txt')
                with open(sample_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        sample_id = int(line.strip())
                        self.samples.append((folder, sample_id))
            else:
                sample_path = os.path.join(folder_path, 'test.txt')
                with open(sample_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        sample_id = int(line.strip())
                        self.samples.append((folder, sample_id))
            

    def load_metadata(self):
        for obj_id in self.objlist:
            model_path = os.path.join(self.dataset_root, 'models', f"obj_{obj_id:02d}.ply")
            self.pt[obj_id] = self.load_model_points(model_path)
    #Define here some usefull functions to access the data
    def load_image(self, img_path):
        # Load an RGB image and convert to tensor.
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
    def load_depth(self, depth_path):
        # Load a depth image and convert to tensor.
        depth = Image.open(depth_path)
        return np.array(depth)
    def load_mask(self, mask_path):
        # Load a mask image and convert to tensor.
        mask = Image.open(mask_path).convert("RGBA")
        return self.transform(mask)
    def load_pose(self, pose_path, idx):
        # Load a 6D pose.
        with open(pose_path, 'r') as f:
            pose_data = yaml.load(f, Loader=yaml.CLoader)
            sample_data = pose_data[idx][0]
            # Convert to numpy array
            rot_mat = np.array(sample_data['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            tras_vec = np.array(sample_data['cam_t_m2c'], dtype=np.float32) / 1000.0
        return rot_mat, tras_vec
    def load_bbx(self, bbx_path, idx):
        # Load a bounding box.
        with open(bbx_path, 'r') as f:
            bbx_data = yaml.load(f, Loader=yaml.CLoader)
            sample_data = bbx_data[idx][0]
            # Convert to numpy array
            bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
        return bbx[0], bbx[1], bbx[2], bbx[3]
    def load_model_points(self, path):
        with open(path) as f:
            assert f.readline().strip() == "ply"
            while f.readline().strip() != "end_header":
                continue
            points = [list(map(float, f.readline().split()[:3])) for _ in range(int(f.readline().split()[-1]))]
        return np.array(points, dtype=np.float32)

    def __len__(self):
        #Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        #print(f"Loading sample {idx}: folder {folder_id}, sample {sample_id}")
        # LOADING PATHS
        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"depth/{sample_id:04d}.png")
        pose_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml") 
        bbx_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")

        # DATA
        img = self.load_image(img_path)
        depth = self.load_depth(depth_path)
        rot_mat, tras_vec = self.load_pose(pose_path, sample_id)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))

        # BB AND MASK EXTRACTION
        if self.split == 'eval':
            # Derive the BB using YOLO
            img_resized = F.interpolate(img.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)
            self.model.eval()
            with torch.no_grad():
                results = self.model.predict(img_resized, conf=0.5, iou=0.5, device='cpu')
            if len(results) == 0:
                raise ValueError(f"No detections found for image {img_path}.")
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            mask = class_ids == (folder_id - 1)
            if not np.any(mask):
                raise ValueError(f"No detections found for object {folder_id} in image {img_path}.")
            boxes = boxes[mask]
            confidences = confidences[mask]
            # Get the bounding box with the highest confidence
            x1, y1, x2, y2  = boxes[np.argmax(confidences)]
            # Rescale to original image dimensions
            orig_H, orig_W = img.shape[1:]
            scale_x = orig_W / 640
            scale_y = orig_H / 640

            x1_orig = int(x1 * scale_x)
            x2_orig = int(x2 * scale_x)
            y1_orig = int(y1 * scale_y)
            y2_orig = int(y2 * scale_y)

            rmin, rmax = int(y1_orig), int(y2_orig)
            cmin, cmax = int(x1_orig), int(x2_orig)
            # Ensure the bounding box is within image dimensions
            rmin, rmax = max(0, rmin), min(480, rmax)
            cmin, cmax = max(0, cmin), min(640, cmax)
            # Mask from SegNet
            mask_path = os.path.join(self.dataset_root, 'segnet_results', f"{folder_id:02d}_label", f"{sample_id:04d}_label.png")
            label = np.array(Image.open(mask_path))
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            # BB from ground truth
            bbx_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"gt.yml")
            rmin, rmax, cmin, cmax = self.get_bbox(self.load_bbx(bbx_path, sample_id))
            # Mask from ground truth
            mask_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"mask/{sample_id:04d}.png")
            label = np.array(Image.open(mask_path))
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        mask = mask_label * mask_depth

        # CROP
        img_crop = img[:, rmin:rmax, cmin:cmax]
        depth_crop = depth[rmin:rmax, cmin:cmax]

        # POINT CLOUD
        cloud, choose = self.sample_points(depth_crop, rmin, rmax, cmin, cmax, mask)

        # DATA AUGMENTATION
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        if self.add_noise:
            cloud = np.add(cloud, add_t)
            img_crop = self.trancolor(img_crop)
        img_crop = self.norm(img_crop)

        # MODEL POINTS
        model_points = self.pt[folder_id] / 1000.0
        model_points = self.sample_model_points(model_points)

        # TARGET POSE AND POINTS
        target_r = rot_mat
        target_t = tras_vec
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        return {
            "rgb": img,
            'depth': torch.from_numpy(depth.astype(np.float32)),
            'cloud': torch.from_numpy(cloud.astype(np.float32)),
            'choose': torch.from_numpy(choose.astype(np.int32)),
            'image': img_crop,
            'target': torch.from_numpy(target.astype(np.float32)),
            'model_points': torch.from_numpy(model_points.astype(np.float32)),
            'obj_id': torch.tensor(folder_id, dtype=torch.int64)
        }
    
    def plotitem(self, idx, show=True):
        img, depth, cloud, choose, image, target, model_points, obj_id = self.__getitem__(idx).values()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        unnorm_img = unnormalize(image, mean, std)
        img_np = unnorm_img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        fix, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].imshow(img.permute(1, 2, 0))
        ax[0].set_title("RGB Image")
        ax[1].imshow(depth.cpu().numpy(), cmap='gray')
        ax[1].set_title("Depth Image")
        ax[2].imshow(img_np)
        ax[2].set_title("YOLO Output")
        # Plot 3D point cloud
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=cloud[:, 0],
            y=cloud[:, 1],
            z=cloud[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.8
            )
        ))
        fig.add_trace(go.Scatter3d(
            x=target[:, 0],
            y=target[:, 1],
            z=target[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.8
            )
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title=f"3D Point Cloud (Object ID: {obj_id.item()})"
        )
        fig.show()
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

    def get_bbox(self, bbox):
        # Convert the bounding box to pixel coordinates
        x, y, width, height = bbox
        rmin = int(y)
        rmax = int(y + height)
        cmin = int(x)
        cmax = int(x + width)

        # Ensure the bounding box is within image dimensions
        rmin, rmax = max(0, rmin), min(self.img_width, rmax)
        cmin, cmax = max(0, cmin), min(self.img_length, cmax)

        return rmin, rmax, cmin, cmax
    
    def sample_points(self, depth, rmin, rmax, cmin, cmax, mask):
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            choose = np.zeros(self.num_points, dtype=np.int32)
        elif len(choose) > self.num_points:
            # TODO: choose a better sampling method than random
            choose = np.random.choice(choose, self.num_points, replace=False)
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

        depth_masked = depth.flatten()[choose][:, np.newaxis]
        xmap = np.array([[j for i in range(640)] for j in range(480)])
        ymap = np.array([[i for i in range(640)] for j in range(480)])
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]

        pt2 = depth_masked / 1000.0
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        return cloud, choose

    def sample_model_points(self, model_points):
        if len(model_points) > self.num_points:
            indices = np.random.choice(len(model_points), self.num_points, replace=False)
            return model_points[indices]
        return model_points
    
    def get_sym_list(self):
        return self.symmetry_obj_idx
    
    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

if __name__ == '__main__':
    # CHECK FOR DATASET POSITION
    dataset_root = os.path.join(os.path.dirname(__file__), '../../dataset/linemod/DenseFusion/Linemod_preprocessed/')
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset not found at {dataset_root}. Please check the path.")
    print(f"Dataset found at {dataset_root}.")

    # DATASET TEST
    train_dataset = PoseDataset(
        dataset_root=dataset_root,
        split='eval',
        train_ratio=0.8,
        seed=42,
    )
    # DATASET PLOT TEST:
    idx = 0
    train_dataset.plotitem(idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


    print(f"Training samples: {len(train_dataset)}")