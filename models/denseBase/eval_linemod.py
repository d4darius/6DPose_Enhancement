import _init_paths
import argparse
import os
import numpy as np
import yaml
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataload.dataloader import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import wandb

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

#--------------------------------------------------------
# ARGUMENT PARSING: Setup the argument for training
#--------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--workers', type=int, default = 2, help='number of data loading workers')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--refine_start', type=bool, default = False, help='whether to start with refinement')
parser.add_argument('--num_points', type=int, default = 500, help='number of points to sample')

opt = parser.parse_args()

# Initialize W&B
wandb.init(
    project="6D-Pose-Estimation-Eval",  # Replace with your project name
    config={
        "dataset_root": opt.dataset_root,
        "model": opt.model,
        "refine_model": opt.refine_model,
        "num_points": opt.num_points,
        "refinement_iterations": 4,
    }
)

def main():
    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 500
    iteration = 4
    bs = 1
    #--------------------------------------------------------
    # DATASET INITIALIZATION: Setup the dataset
    #--------------------------------------------------------
    dataset_config_dir = '../dataset/linemod/DenseFusion/Linemod_Preprocessed/models'
    output_result_dir = 'experiments/eval_result/linemod'

    #--------------------------------------------------------
    # MODEL INITIALIZATION: Setup the estimator and refiner models
    #--------------------------------------------------------
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.to(device)
    if not(os.path.exists(opt.model)):
        print('File not found: {0}'.format(opt.model))
        exit(0)
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()
    if os.path.exists(opt.refine_model):
        opt.refine = True
        refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
        refiner.to(device)
        refiner.load_state_dict(torch.load(opt.refine_model))
        refiner.eval()
    else:
        opt.refine = False

    testdataset = PoseDataset_linemod(opt.dataset_root, 'eval', num_points=opt.num_points, add_noise=False, noise_trans=0.0, refine=opt.refine_start, device=device)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)
    if opt.refine:
        criterion_refine = Loss_refine(num_points_mesh, sym_list)

    diameter = []
    if not os.path.exists(dataset_config_dir):
        print('Please set the correct dataset config dir!')
        exit(0)
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
    meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
    meta = yaml.safe_load(meta_file)
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print(diameter)

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

    #--------------------------------------------------------
    # EVALUATION LOOP: Loop through the test dataset
    #--------------------------------------------------------
    for i, data in enumerate(testdataloader, 0):
        if len(data) == 1:
            print(data['error'])
            continue
        points = data['cloud']
        choose = data['choose']
        img = data['image']
        target = data['target']
        model_points = data['model_points']
        idx = data['obj_id']
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
            continue
        points, choose, img, target, model_points, idx = Variable(points).to(device), \
                                                        Variable(choose).to(device), \
                                                        Variable(img).to(device), \
                                                        Variable(target).to(device), \
                                                        Variable(model_points).to(device), \
                                                        Variable(idx).to(device)

        #--------------------------------------------------------
        # POSE INFERENCE: Estimate the initial pose using PoseNet
        #--------------------------------------------------------
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        #--------------------------------------------------------
        # REFINEMENT LOOP: Refine the pose using PoseRefineNet
        #--------------------------------------------------------
        if opt.refine:
            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).to(device).view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).to(device).view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_points = torch.bmm((points - T), R).contiguous()
                pred_r, pred_t = refiner(new_points, emb, idx)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)
                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

        # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

        #--------------------------------------------------------
        # RESULTS: Evaluate the pose estimation result for each object
        #--------------------------------------------------------
        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()

        #--------------------------------------------------------
        # DISTANCE MEASURE: Calculate the distance between the estimated and ground truth poses
        #--------------------------------------------------------
        if idx[0].item() in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).to(device)
            target = torch.from_numpy(target.astype(np.float32)).to(device)
            pred_exp = pred.unsqueeze(1)
            target_exp = target.unsqueeze(0)
            dists = torch.norm(pred_exp - target_exp, dim=2)
            inds = dists.argmin(dim=1)
            target_matched = target[inds]
            dis = torch.mean(torch.norm(pred - target_matched, dim=1)).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))

        # Log metrics to W&B
        wandb.log({
            "object_id": idx[0].item(),
            "distance": dis,
            "success": dis < diameter[idx[0].item()],
            "iteration": i,
        })

        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
            print('No.{0} Pass! Distance: {1}'.format(i, dis))
            fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
        else:
            print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
            fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
        num_count[idx[0].item()] += 1

    for i in range(num_objects):
        print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
        fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
    print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    fw.close()

    # Save evaluation logs to W&B
    wandb.save('{0}/eval_result_logs.txt'.format(output_result_dir))

if __name__ == '__main__':
    main()