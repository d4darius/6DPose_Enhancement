import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy.ma as ma
import torch
from ultralytics import YOLO
import open3d as o3d
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def to_graph_data(cloud, k=6):
    cloud_tensor = torch.from_numpy(cloud.astype(np.float32))
    num_nodes = cloud_tensor.size(0)
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.arange(num_nodes).repeat_interleave(num_nodes)
    mask = row != col  # Remove self-loops if desired
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    data = Data(pos=cloud_tensor, edge_index=edge_index)
    return data

class PoseDataset(Dataset):
    def __init__(self, dataset_root, split='train', split_ratio=0.7, seed=42, num_points=500, add_noise=False, noise_trans=0.03, refine=False, device='cpu', sampling='random'):

        self.dataset_root = dataset_root
        self.split = split
        self.seed = seed
        self.num_points = num_points
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.refine = refine
        self.device = device
        self.split_ratio = split_ratio
        self.sampling = sampling

        # Object list and metadata
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.obj_id_map = {real_id: i for i, real_id in enumerate(self.objlist)}
        self.pt = {}
        self.gt_data_cache = {} # New: Cache for gt.yml data

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
        self.symmetry_obj_idx = [10, 11]

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.args = os.path.join(os.path.dirname(__file__), '../models/yolo/weights/best.pt')
        self.model = YOLO(self.args)

        self.samples = []
        self.load_dataset()
        self.load_metadata()
        self.preload_gt_data()

        # Precompute full xmap and ymap
        self.xmap_full = np.array([[j for _ in range(self.img_length)] for j in range(self.img_width)])
        self.ymap_full = np.array([[i for i in range(self.img_length)] for _ in range(self.img_width)])
        
    def load_dataset(self):
        names = [int(n.split("/")[-1]) for n in sorted([f for f in os.listdir(os.path.join(self.dataset_root, 'data')) if os.path.isdir(os.path.join(self.dataset_root, 'data', f))])]

        for folder in names:
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder:02d}")
            sample_codes = [int(n.split("/")[-1].split(".")[0]) for n in sorted([f for f in os.listdir(os.path.join(folder_path, 'rgb')) if f.endswith('.png')])]
            train_samples, test_samples = train_test_split(sample_codes, 
                                                          test_size=1-self.split_ratio, 
                                                          random_state=self.seed, 
                                                          shuffle=True)
            if self.split == 'train':
                for sample_id in train_samples:
                    self.samples.append((folder, sample_id))
            else:
                for sample_id in test_samples:
                    self.samples.append((folder, sample_id))

    def preload_gt_data(self):
        print("Preloading ground truth YAML data...")
        folder_ids = sorted(list(set(sample[0] for sample in self.samples))) # Get unique folder_ids
        for folder_id in folder_ids:
            gt_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "gt.yml")
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    self.gt_data_cache[folder_id] = yaml.load(f, Loader=yaml.CLoader)
            else:
                print(f"Warning: gt.yml not found for folder {folder_id}")
        print("Finished preloading ground truth YAML data.")
    
    def load_model_points(self, path):
        points = []
        with open(path) as f:
            # Check first line is "ply"
            assert f.readline().strip() == "ply"
            
            # Parse header to find vertex count
            vertex_count = None
            while True:
                line = f.readline().strip()
                if line == "end_header":
                    break
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
            
            # Read vertices
            if vertex_count is not None:
                for _ in range(vertex_count):
                    # Take first 3 values from each line (x, y, z)
                    points.append(list(map(float, f.readline().split()[:3])))
            
        return np.array(points, dtype=np.float32)

    def load_metadata(self):
        for obj_id in self.objlist:
            model_path = os.path.join(self.dataset_root, 'models', f"obj_{obj_id:02d}.ply")
            self.pt[obj_id] = self.load_model_points(model_path)
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
    def load_pose(self, folder_id, sample_id_in_yaml):
        # Load a 6D pose from cached data
        if folder_id not in self.gt_data_cache:
            # This should ideally not happen if preload_gt_data worked
            raise ValueError(f"gt.yml data for folder {folder_id} not found in cache.")
        
        pose_data_for_folder = self.gt_data_cache[folder_id]
        
        if sample_id_in_yaml not in pose_data_for_folder:
             raise ValueError(f"Sample ID {sample_id_in_yaml} not found in gt.yml for folder {folder_id}")

        sample_data_list = pose_data_for_folder[sample_id_in_yaml]
        if not sample_data_list: # Check if the list is empty
            raise ValueError(f"No pose data for sample ID {sample_id_in_yaml} in folder {folder_id}")
        
        for data in sample_data_list:
            if data['obj_id'] == folder_id:
                sample_data = data
                break
        rot_mat = np.array(sample_data['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        tras_vec = np.array(sample_data['cam_t_m2c'], dtype=np.float32) / 1000.0
        return rot_mat, tras_vec

    def load_bbx(self, folder_id, sample_id_in_yaml):
        # Load a bounding box from cached data
        if folder_id not in self.gt_data_cache:
            raise ValueError(f"gt.yml data for folder {folder_id} not found in cache.")
        
        bbx_data_for_folder = self.gt_data_cache[folder_id]

        if sample_id_in_yaml not in bbx_data_for_folder:
             raise ValueError(f"Sample ID {sample_id_in_yaml} not found in gt.yml for folder {folder_id}")
        
        sample_data_list = bbx_data_for_folder[sample_id_in_yaml]
        if not sample_data_list:
             raise ValueError(f"No bounding box data for sample ID {sample_id_in_yaml} in folder {folder_id}")

        for data in sample_data_list:
            if data['obj_id'] == folder_id:
                sample_data = data
                break
        bbx = np.array(sample_data['obj_bb'], dtype=np.float32)
        return bbx[0], bbx[1], bbx[2], bbx[3]

            
    #Define here some usefull functions to access the data
    def __len__(self):
        #Return the total number of samples in the selected split.
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx] # sample_id here is the file name like 0000, 0001
        mapped_id = self.obj_id_map[folder_id]

        # LOADING PATHS (only for non-YAML data now)
        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"depth/{sample_id:04d}.png")
        
        if not os.path.exists(img_path):
                return {
                    "error": f"Mask file not found: {img_path}"
                }
        if not os.path.exists(depth_path):
                return {
                    "error": f"Mask file not found: {depth_path}"
                }

        # DATA
        img = self.load_image(img_path)
        depth = self.load_depth(depth_path)
        
        # Use sample_id as the key for the YAML data, which seems to be 0-indexed in the YAML file
        # The sample_id from self.samples is the file name (e.g., 0, 1, 2, ... for 0000.png, 0001.png)
        yaml_key_sample_id = sample_id 

        try:
            rot_mat, tras_vec = self.load_pose(folder_id, yaml_key_sample_id)
        except ValueError as e:
            return {"error": f"Error loading pose for folder {folder_id}, sample {sample_id} (yaml key {yaml_key_sample_id}): {e}"}

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))

        # BB AND MASK EXTRACTION
        if self.split == 'eval':
            # Derive the BB using YOLO
            img_resized = F.interpolate(img.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)
            self.model.eval()
            with torch.no_grad():
                results = self.model.predict(img_resized, conf=0.5, iou=0.5, device=self.device, verbose=False)
            if len(results[0].boxes) == 0:
                return {
                        "error": "No detection found"
                    }
            else:
                result = results[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                mask = class_ids == (folder_id - 1)
                if not np.any(mask):
                    return {
                        "error": "No detection found"
                    }
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
            if not os.path.exists(mask_path):
                return {
                    "error": f"Mask file not found: {mask_path}"
                }
            else:
                label = np.array(Image.open(mask_path))
                mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255))) # Assuming 255 is object
        else: # 'train' split
            try:
                # BB from ground truth (cached)
                x, y, w, h = self.load_bbx(folder_id, yaml_key_sample_id)
                rmin, rmax, cmin, cmax = self.get_bbox([x,y,w,h]) # Pass as a list/tuple
            except ValueError as e:
                 return {"error": f"Error loading bbx for folder {folder_id}, sample {sample_id} (yaml key {yaml_key_sample_id}): {e}"}

            # Mask from ground truth
            mask_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"mask/{sample_id:04d}.png")
            if not os.path.exists(mask_path):
                return {
                    "error": f"Mask file not found: {mask_path}"
                }
            else:
                label = np.array(Image.open(mask_path))
                # Ensure mask_label is boolean and 2D
                if label.ndim == 3 and label.shape[2] >= 3: # Check if it's an RGB-like mask
                     mask_label = np.all(label == [255,255,255], axis=2)
                elif label.ndim == 2: # Grayscale mask
                     mask_label = (label == 255) # Assuming 255 is the object
                else:
                     return {"error": f"Unexpected mask format for {mask_path}"}
                mask_label = ma.getmaskarray(ma.masked_equal(mask_label, True))

        # Ensure rmin, rmax, cmin, cmax are defined before this point for both splits
        if 'rmin' not in locals() or 'rmax' not in locals() or 'cmin' not in locals() or 'cmax' not in locals():
             return {"error": f"Bounding box coordinates not defined for folder {folder_id}, sample {sample_id}"}

        mask = mask_label * mask_depth

        # CROP
        img_crop = img[:, rmin:rmax, cmin:cmax]
        depth_crop = depth[rmin:rmax, cmin:cmax]

        # POINT CLOUD
        cloud, choose = self.sample_points(depth_crop, rmin, rmax, cmin, cmax, mask)
        graph_data = to_graph_data(cloud)

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
            'graph': graph_data,
            'choose': torch.from_numpy(choose.astype(np.int32)),
            'image': img_crop,
            'target': torch.from_numpy(target.astype(np.float32)),
            'model_points': torch.from_numpy(model_points.astype(np.float32)),
            'obj_id': torch.tensor(mapped_id, dtype=torch.int64)
        }
    
    def getObjectId(self, idx):
        folder_id, sample_id = self.samples[idx]
        mapped_id = self.obj_id_map[folder_id]
        return torch.tensor(mapped_id, dtype=torch.int64)
    
    def plotitem(self, idx, show=True):
        data = self.__getitem__(idx)
        if len(data) == 1:
            print(data['error'])
            return
        image = data['image']
        depth = data['depth']
        cloud = data['cloud']
        choose = data['choose']
        img = data['rgb']
        target = data['target']
        model_points = data['model_points']
        obj_id = data['obj_id']
        graph = data['graph']

        # Convert to NetworkX graph
        G = to_networkx(graph, to_undirected=True)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        unnorm_img = unnormalize(image, mean, std)
        img_np = unnorm_img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        fix, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(img.permute(1, 2, 0))
        ax[0].set_title("RGB Image")
        ax[1].imshow(depth.cpu().numpy(), cmap='gray')
        ax[1].set_title("Depth Image")
        ax[2].imshow(img_np)
        ax[2].set_title("YOLO Output")
        pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.pos.size(0))}
        nx.draw(G, pos, ax=ax[3], with_labels=False, node_size=10, node_color='red', edge_color='black')
        ax[3].set_title("Graph (Top-down)")
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
    
    def sample_points(self, depth_crop, rmin, rmax, cmin, cmax, mask_full):
        # depth_crop is the depth image already cropped to the bounding box [rmin:rmax, cmin:cmax]
        # mask_full is the boolean mask for the entire image (e.g., mask_label * mask_depth)

        # 1. Get the mask for the cropped region
        mask_cropped = mask_full[rmin:rmax, cmin:cmax]
        
        # 2. Get indices of valid points within the flattened cropped mask
        # These indices are relative to the flattened mask_cropped / depth_crop
        initial_choose = mask_cropped.flatten().nonzero()[0]
        num_available_points = len(initial_choose)

        final_choose_indices = None # This will store the self.num_points indices

        if num_available_points == 0:
            # No points in the mask, final_choose_indices will be all zeros.
            # These indices will be applied to flattened depth_crop, xmap_crop, ymap_crop.
            final_choose_indices = np.zeros(self.num_points, dtype=np.int32)
        
        elif num_available_points > self.num_points:
            if self.sampling == 'random':
                # Randomly select self.num_points indices from the indices of available points
                idx_into_initial_choose = np.random.choice(num_available_points, self.num_points, replace=False)
                final_choose_indices = initial_choose[idx_into_initial_choose]
            
            elif self.sampling == 'FPS' or self.sampling == 'curvature':
                # Build a temporary cloud from all available points in initial_choose
                # to perform FPS or curvature sampling on.
                xmap_for_crop_sampling = self.xmap_full[rmin:rmax, cmin:cmax]
                ymap_for_crop_sampling = self.ymap_full[rmin:rmax, cmin:cmax]

                depth_values_for_sampling = depth_crop.flatten()[initial_choose][:, np.newaxis]
                xmap_values_for_sampling = xmap_for_crop_sampling.flatten()[initial_choose][:, np.newaxis]
                ymap_values_for_sampling = ymap_for_crop_sampling.flatten()[initial_choose][:, np.newaxis]

                pt2_sampling = depth_values_for_sampling / 1000.0
                pt0_sampling = (ymap_values_for_sampling - self.cam_cx) * pt2_sampling / self.cam_fx
                pt1_sampling = (xmap_values_for_sampling - self.cam_cy) * pt2_sampling / self.cam_fy
                cloud_for_sampling = np.concatenate((pt0_sampling, pt1_sampling, pt2_sampling), axis=1)
                
                # Sample self.num_points from this cloud_for_sampling.
                # The returned indices (selected_sub_indices) are indices into cloud_for_sampling,
                # and therefore also indices into the initial_choose array.
                if self.sampling == 'FPS':
                    selected_sub_indices = self.FPS_point_sampling(cloud_for_sampling, self.num_points)
                elif self.sampling == 'curvature':
                    selected_sub_indices = self.curvature_point_sampling(cloud_for_sampling, self.num_points)
                else:
                    return None # Invalid sampling method
                
                # The final chosen indices are the original mask indices corresponding to these sampled points
                final_choose_indices = initial_choose[selected_sub_indices]
        
        else: # 0 < num_available_points <= self.num_points
            # Not enough points, pad with existing points (wrap around)
            final_choose_indices = np.pad(initial_choose, (0, self.num_points - num_available_points), 'wrap')

        # Now, construct the final cloud using final_choose_indices.
        # These indices are relative to the flattened cropped region (depth_crop, xmap_at_crop, ymap_at_crop).
        
        xmap_at_crop = self.xmap_full[rmin:rmax, cmin:cmax]
        ymap_at_crop = self.ymap_full[rmin:rmax, cmin:cmax]

        depth_masked = depth_crop.flatten()[final_choose_indices][:, np.newaxis]
        xmap_masked = xmap_at_crop.flatten()[final_choose_indices][:, np.newaxis]
        ymap_masked = ymap_at_crop.flatten()[final_choose_indices][:, np.newaxis]

        pt2 = depth_masked / 1000.0
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx # X coordinate in camera frame
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy # Y coordinate in camera frame
        final_cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        
        return final_cloud, final_choose_indices
    
    def curvature_point_sampling(self, cloud, num_keypoints):
        #Select 3D keypoints based on curvature and spatial distribution.
        
        
        # Estimate normals
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(cloud)
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        
            # Compute curvature using eigenvalues of the covariance matrix
        curvatures = []
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        points = np.asarray(point_cloud.points)

        for i in range(len(points)):
            # Find neighbors
            _, idx, _ = kdtree.search_knn_vector_3d(point_cloud.points[i], 30)  # Use 30 nearest neighbors
            neighbors = points[idx, :]
        
            # Compute covariance matrix
            covariance = np.cov(neighbors.T)
        
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(covariance)
            eigenvalues = np.sort(eigenvalues)  # Ensure eigenvalues are sorted
        
            # Curvature is the smallest eigenvalue divided by the sum of all eigenvalues
            curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-6)  # Add small value to avoid division by zero
            curvatures.append(curvature)

        curvatures = np.array(curvatures)
        curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))  # Normalize


        # Select keypoints based on curvature and spatial distribution
        keypoints = []
        for _ in range(num_keypoints):
            if len(keypoints) == 0:
                idx = np.argmax(curvatures)  # Start with the point with the highest curvature
            else:
                distances = np.linalg.norm(np.asarray(point_cloud.points) - np.asarray(point_cloud.points)[keypoints[-1]], axis=1)
                weights = curvatures - 0.5 * distances  # Weighted combination of curvature and distance
                idx = np.argmax(weights)
            keypoints.append(idx)
            curvatures[idx] = 0  # Avoid selecting the same point again
        
        return np.array(keypoints)
    
    def FPS_point_sampling(self, cloud, num_keypoints):
        # Select keypoints using farthest point sampling
        if len(cloud) == 0:
            return np.array([])

        # Convert to numpy array if not already
        points = np.asarray(cloud)
        N = points.shape[0]
        if num_keypoints >= N:
            return np.arange(N)

        keypoints = [np.random.randint(N)]
        distances = np.full(N, np.inf)
        for _ in range(1, num_keypoints):
            last = points[keypoints[-1]]
            dist = np.linalg.norm(points - last, axis=1)
            distances = np.minimum(distances, dist)
            next_idx = np.argmax(distances)
            keypoints.append(next_idx)
        return np.array(keypoints)

    def sample_model_points(self, model_points):
        if len(model_points) >= self.num_points:
            indices = np.random.choice(len(model_points), self.num_points, replace=False)
            return model_points[indices]
        else:
            # Not enough points — pad using wrap-around
            repeat_factor = int(np.ceil(self.num_points / len(model_points)))
            padded = np.tile(model_points, (repeat_factor, 1))
            return padded[:self.num_points]

    
    def get_sym_list(self):
        return self.symmetry_obj_idx
    
    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small
    
    @staticmethod
    def center_pad_collate(batch):
        """
        Custom collate function that pads images to a uniform size while keeping them centered,
        and adjusts the 'choose' indices to account for padding.
        """
        # Filter out error samples
        batch = [sample for sample in batch if 'error' not in sample]
        if not batch:
            return {'error': 'Empty batch after filtering'}
        
        # Find max dimensions in this batch
        max_h = max([item['image'].shape[1] for item in batch])
        max_w = max([item['image'].shape[2] for item in batch])
        
        # Round up to next multiple of 8 for predicting the PSPnet output size
        max_h = ((max_h + 7) // 8) * 8
        max_w = ((max_w + 7) // 8) * 8
        
        # Prepare the output dictionary
        batch_dict = {k: [] for k in batch[0].keys()}
        pad_info = []  # Store padding info for debugging
        
        for item in batch:
            # Get original dimensions
            img = item['image']
            c, h, w = img.shape
            
            # Calculate padding for centering
            pad_h_top = (max_h - h) // 2
            pad_h_bottom = max_h - h - pad_h_top
            pad_w_left = (max_w - w) // 2
            pad_w_right = max_w - w - pad_w_left
            
            # Pad image (keep centered)
            padded_img = F.pad(img, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))
            batch_dict['image'].append(padded_img)
            
            # Update choose indices to account for padding
            choose = item['choose'].clone()
            
            # Convert flattened indices to 2D coordinates in original crop
            y_orig = choose // w
            x_orig = choose % w
            
            # Adjust coordinates for padding
            y_padded = y_orig + pad_h_top
            x_padded = x_orig + pad_w_left
            
            # Convert back to flattened indices in padded image
            padded_choose = y_padded * max_w + x_padded
            
            batch_dict['choose'].append(padded_choose)
            pad_info.append({
                'original_shape': (h, w),
                'padded_shape': (max_h, max_w),
                'padding': (pad_h_top, pad_h_bottom, pad_w_left, pad_w_right)
            })
            
            # Add all other tensors
            for k in item.keys():
                if k not in ['image', 'choose']:
                    batch_dict[k].append(item[k])
        
        # Stack tensors
        for k in batch_dict:
            try:
                batch_dict[k] = torch.stack(batch_dict[k])
            except:
                # Keep as list if can't be stacked
                pass
        
        # same for graph
        batch_dict['graph'] = Batch.from_data_list([d['graph'] for d in batch])
        
        # Add padding information for debugging/verification
        batch_dict['pad_info'] = pad_info
        return batch_dict

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
        seed=42,
        sampling='random',  # or 'random' or 'FPS'
    )
    # DATASET PLOT TEST:
    idx = 3050
    train_dataset.plotitem(idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


    print(f"Training samples: {len(train_dataset)}")