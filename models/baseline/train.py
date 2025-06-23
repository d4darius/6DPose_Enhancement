import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch.backends.cudnn as cudnn
from dataload.dataloader import PoseDataset as PoseDataset_linemod
from lib.network import PosePredMLP
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import yaml

#--------------------------------------------------------
# Configuration
#--------------------------------------------------------
dataset_root = os.path.join(os.path.dirname(__file__), '../../dataset/linemod/DenseFusion/Linemod_preprocessed/')
num_points = 500
epochs = 100
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

#upload diameters
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
diameter = []

meta_file = open('{0}/models_info.yml'.format(os.path.join(dataset_root, "models")), 'r')
meta = yaml.safe_load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)



#--------------------------------------------------------
# Utility functions
#--------------------------------------------------------
def extract_rgb(item):
    image_crop = item['image']  # [3, H, W]
    choose = item['choose']
    h, w = image_crop.shape[1], image_crop.shape[2]

    if choose.max() >= h * w:
        raise IndexError("choose index out of bounds")

    img_np = image_crop.permute(1, 2, 0).cpu().numpy()
    img_flat = img_np.reshape(-1, 3)
    return img_flat[choose.cpu().numpy()]

def rot_to_quaternion(rot_mat):
    return R.from_matrix(rot_mat).as_quat()

def loader(bigDataset, valid_count = 0, num_points = num_points, input_dim = 6, target_dim = 3):
    max_samples = len(bigDataset)

    X_np = np.zeros((max_samples, num_points, input_dim), dtype=np.float32) # [N, 500, 6] xyz + rgb
    y_np = np.zeros((max_samples, num_points, target_dim), dtype=np.float32)
    obj_id_np = np.zeros((max_samples,), dtype=np.int64)


    model_points_dict = {}

    for i in tqdm(range(len(bigDataset)), desc="Preparazione dataset"):
        item = bigDataset[i]
        if 'error' in item:
            continue
        try:
            cloud = item['cloud'].cpu().numpy()
            rgb = extract_rgb(item)
            features = np.concatenate([cloud, rgb], axis=1)

            target = item['target'].cpu().numpy()

            X_np[valid_count] = features
            y_np[valid_count] = target

            obj_id = item['obj_id']
            obj_id_np[valid_count] = obj_id
            obj_id = obj_id.item()
            
            if obj_id not in model_points_dict:
                model_points_dict[obj_id] = bigDataset[valid_count]['model_points']
                                                                        
            valid_count += 1

        except Exception as e:
            print(f"[{i}] Skipped: {e}")
            continue

    X_np = X_np[:valid_count]
    y_np = y_np[:valid_count]
    obj_id_np = obj_id_np[:valid_count]

    return X_np, y_np, obj_id_np, model_points_dict


def pose_loss(pred, target, obj_ids, model_points_dict):
    B = pred.size(0)
    device = pred.device
    total_loss = 0.0
    mean_target = target.mean(dim=1)

    for i in range(B):
        obj_id = obj_ids[i].item()
        model_pts = model_points_dict[obj_id].to(device)

        pred_q = pred[i, 3:].unsqueeze(0)
        pred_t = pred[i, :3].unsqueeze(0)

        pred_r = quaternion_to_rotation_matrix(pred_q)[0]

        pred_transformed = model_pts @ pred_r.T + pred_t

        mse = F.mse_loss(pred_transformed, mean_target[i].unsqueeze(0))
        total_loss += mse

    return total_loss / B

def quaternion_to_rotation_matrix(q):
    if q.dim() == 1:
        q = q.unsqueeze(0)

    B = q.shape[0]
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]

    R = torch.zeros((B, 3, 3), device=q.device)
    R[:,0,0] = 1 - 2*(y**2 + z**2)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)

    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x**2 + z**2)
    R[:,1,2] = 2*(y*z - x*w)

    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x**2 + y**2)

    return R

# ADD matrix computation
def compute_add(pred_pose, gt_pose, model_points):

    B = pred_pose.shape[0]
    N = model_points.shape[0]

    pred_rot = quaternion_to_rotation_matrix(pred_pose[:, 3:])
    pred_trans = pred_pose[:, :3].unsqueeze(1)

    model_points = torch.tensor(model_points, dtype=torch.float32, device=pred_pose.device)
    model_points = model_points.unsqueeze(0).expand(B, -1, -1)

    pred_transformed = torch.bmm(model_points, pred_rot.transpose(1,2)) + pred_trans

    add = torch.norm(pred_transformed - gt_pose, dim=2).mean(dim=1)

    return add

class PoseRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, X_np, y_np, obj_id_np):
        self.X_np = X_np
        self.y_np = y_np
        self.obj_id = obj_id_np

    def __len__(self):
        return len(self.X_np)

    def __getitem__(self, idx):
        X = torch.tensor(self.X_np[idx], dtype=torch.float32)
        y = torch.tensor(self.y_np[idx], dtype=torch.float32)
        return X, y, self.obj_id[idx]
    

# --------------------------------------------------------
# Train loop
# --------------------------------------------------------
bigDataset = PoseDataset_linemod(dataset_root, 'train', num_points=num_points, add_noise=True, device=device, sampling='random')
X_np, y_np, obj_id_np, model_points_dict = loader(bigDataset)
dataset = PoseRegressionDataset(X_np, y_np, obj_id_np)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


model = PosePredMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch, obj_ids_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        obj_ids_batch = obj_ids_batch.to(device)

        pred = model(X_batch)
        loss = pose_loss(pred, y_batch, obj_ids_batch, model_points_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.8f}")



#--------------------------------------------------------
# Evaluation
#--------------------------------------------------------
bigTestDataset = PoseDataset_linemod(dataset_root, 'test', num_points=num_points, add_noise=True, device=device, sampling='random')
X_np, y_np, obj_id_np, model_points_dict = loader(bigTestDataset)
dataset = PoseRegressionDataset(X_np, y_np, obj_id_np)
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)


model.eval()
all_add = []
correct = 0
total = 0

add_per_object = defaultdict(list)
correct_per_object = defaultdict(int)
total_per_object = defaultdict(int)

with torch.no_grad():
    for X_batch, y_batch, obj_ids_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)

        batch_size = X_batch.size(0)
        batch_add = []
        mean_target = y_batch.mean(dim=1)

        for i in range(batch_size):
            obj_id = obj_ids_batch[i].item()
            model_pts = model_points_dict[obj_id].to(device)

            add_i = compute_add(pred[i].unsqueeze(0), mean_target[i], model_pts)
            add_value = add_i.item()
            batch_add.append(add_value)

            add_per_object[obj_id].append(add_value)

        all_add.extend(batch_add)


tot_acc = 0
# Per object results
print(f"=== Per object results ===")
for obj_id in sorted(add_per_object.keys()):
    obj_adds = add_per_object[obj_id]
    mean_obj_add = sum(obj_adds) / len(obj_adds)

    obj_threshold = diameter[obj_id]
    correct_count = sum(1 for add in obj_adds if add < obj_threshold)
    acc_obj = correct_count / len(obj_adds)
    tot_acc += acc_obj

    print(f"Obj {obj_id:02d}: Mean ADD = {mean_obj_add:.4f}, Threshold = {obj_threshold:.4f}, Accuracy = {acc_obj*100:.2f}%")
    
# Global results
mean_add = sum(all_add) / len(all_add)
tot_acc = tot_acc / len(diameter) * 100
print(f"\n=== Global results ===")
print(f"Test ADD mean: {mean_add:.4f}")
print(f"Test accuracy mean: {tot_acc:.2f}%")


# Save model
torch.save(model.state_dict(), '{}/baseline_model_{}_{:.4f}.pth'.format('checkpoints/linemod', epoch, mean_add))