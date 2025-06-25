import _init_paths
import argparse
import os
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from dataload.dataloader import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, GNNPoseNet
from lib.loss import Loss
from lib.transformations import quaternion_matrix

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
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--num_points', type=int, default = 500, help='number of points to sample')
parser.add_argument('--gnn', action='store_true', default=False, help='start training on the geometric model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA (use CPU only)')
parser.add_argument('--feat', type=str, default = 'color',  help='resume PoseNet model')

opt = parser.parse_args()

def main():
    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 500
    
    # Force CPU if requested
    use_cuda = torch.cuda.is_available() and not opt.no_cuda
    device_str = "cuda" if use_cuda else "cpu"
    print(f"Using device: {device_str}")
    
    #--------------------------------------------------------
    # DATASET INITIALIZATION: Setup the dataset
    #--------------------------------------------------------
    dataset_config_dir = '../dataset/linemod/DenseFusion/Linemod_preprocessed/models'
    output_result_dir = 'experiments/eval_result/linemod'

    #--------------------------------------------------------
    # MODEL INITIALIZATION: Setup the estimator
    #--------------------------------------------------------
    if opt.gnn:
        estimator = GNNPoseNet(num_points = num_points, num_obj = num_objects)
        estimator.to(device)
        if not(os.path.exists(opt.model)):
            print('File not found: {0}'.format(opt.model))
            exit(0)
        estimator.load_state_dict(torch.load(opt.model, map_location=torch.device(device)))
        estimator.eval()
    else:
        estimator = PoseNet(num_points = num_points, num_obj = num_objects)
        estimator.to(device)
        if not(os.path.exists(opt.model)):
            print('File not found: {0}'.format(opt.model))
            exit(0)
        estimator.load_state_dict(torch.load(opt.model, map_location=torch.device(device)))
        estimator.eval()

    testdataset = PoseDataset_linemod(opt.dataset_root, 'eval', num_points=opt.num_points, add_noise=False, noise_trans=0.0, device=device)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, 
        batch_size=opt.batch_size,
        shuffle=False, 
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=testdataset.center_pad_collate
    )
    
    # Use actual batch size from dataloader
    bs = testdataloader.batch_size
    print(f"Evaluating with batch size: {bs}")
    print(f"Using module with feature {opt.feat}")
    
    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)

    diameter = []
    if not(os.path.exists(dataset_config_dir)):
        print('Please set the correct dataset config dir!')
        exit(0)
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
    meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
    meta = yaml.safe_load(meta_file)
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print("Diameter for each object (in meters):")
    print(diameter)

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    # Add tracking for total distances per object
    total_distances = [0.0 for i in range(num_objects)]
    fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

    #--------------------------------------------------------
    # EVALUATION LOOP: Loop through the test dataset
    #--------------------------------------------------------
    for i, data in enumerate(testdataloader, 0):
        if len(data) == 1:
            print(data['error'][0])
            continue
        points = data['cloud']
        choose = data['choose']
        img = data['image']
        target = data['target']
        model_points = data['model_points']
        if opt.gnn:
            graph_data = data['graph'].to(device)
        idx = data['obj_id']
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
            continue
        points, choose, img, target, model_points, idx = points.to(device), \
                                                                              choose.to(device), \
                                                                              img.to(device), \
                                                                              target.to(device), \
                                                                              model_points.to(device), \
                                                                              idx.to(device)

        # Get the actual batch size from the data
        batch_size = img.size(0)
        
        #--------------------------------------------------------
        # POSE INFERENCE: Estimate the initial pose using PoseNet
        #--------------------------------------------------------
        if opt.gnn:
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, graph_data, idx, opt.feat)
        else:
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(batch_size, num_points, 1)
        pred_c = pred_c.view(batch_size, num_points)
        
        # Process each item in the batch
        for b in range(batch_size):
            how_max, which_max = torch.max(pred_c[b], 0)
            pred_t_b = pred_t.view(batch_size * num_points, 1, 3)
            which_idx = b * num_points + which_max
            
            my_r = pred_r[b][which_max].view(-1).cpu().data.numpy()
            my_t = (points[b][which_max] + pred_t_b[which_idx].view(3)).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            # Here 'my_pred' is the final pose estimation result ('my_r': quaternion, 'my_t': translation)

            #--------------------------------------------------------
            # RESULTS: Evaluate the pose estimation result for each object
            #--------------------------------------------------------
            model_points_b = model_points[b].cpu().detach().numpy()
            my_r_matrix = quaternion_matrix(my_r)[:3, :3]
            # Ensure my_t is properly shaped for broadcasting (3,)
            my_t_reshaped = my_t #if my_t.shape == (3,) else my_t.reshape(3)
            pred = np.dot(model_points_b, my_r_matrix.T) + my_t_reshaped
            target_b = target[b].cpu().detach().numpy()

            #--------------------------------------------------------
            # DISTANCE MEASURE: Calculate the distance between the estimated and ground truth poses
            #--------------------------------------------------------
            obj_idx = idx[b].item()
            if obj_idx in sym_list:
                pred_tensor = torch.from_numpy(pred.astype(np.float32)).to(device)
                target_tensor = torch.from_numpy(target_b.astype(np.float32)).to(device)
                pred_exp = pred_tensor.unsqueeze(1)
                target_exp = target_tensor.unsqueeze(0)
                dists = torch.norm(pred_exp - target_exp, dim=2)
                inds = dists.argmin(dim=1)
                target_matched = target_tensor[inds]
                dis = torch.mean(torch.norm(pred_tensor - target_matched, dim=1)).item()
            else:
                dis = np.mean(np.linalg.norm(pred - target_b, axis=1))

            # Add the distance to the total for this object
            total_distances[obj_idx] += dis
            
            if dis < diameter[obj_idx]:
                success_count[obj_idx] += 1
                print('No.{0} Obj_id {1} Pass! Distance: {2}'.format(i, obj_idx, dis))
                fw.write('No.{0} Obj_id {1} Pass! Distance: {2}\n'.format(i, obj_idx, dis))
            else:
                print('No.{0} Obj_id {1} NOT Pass! Distance: {2}'.format(i, obj_idx, dis))
                fw.write('No.{0} Obj_id {1} NOT Pass! Distance: {2}\n'.format(i, obj_idx, dis))
            num_count[obj_idx] += 1

    # Print individual object success rates and mean distances
    for i in range(num_objects):
        if num_count[i] > 0:
            mean_distance = total_distances[i] / num_count[i]
            print('Object {0} success rate: {1:.4f}, mean distance: {2:.4f}, diameter: {3:.4f}'.format(objlist[i], float(success_count[i]) / num_count[i], mean_distance, diameter[i]))
            fw.write('Object {0} success rate: {1:.4f}, mean distance: {2:.4f}, diameter: {3:.4f}\n'.format(objlist[i], float(success_count[i]) / num_count[i], mean_distance, diameter[i]))
    
    # Overall success rate
    print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    
    # Overall mean distance
    if sum(num_count) > 0:
        overall_mean_distance = sum(total_distances) / sum(num_count)
        print('ALL mean distance: {0}'.format(overall_mean_distance))
        fw.write('ALL mean distance: {0}\n'.format(overall_mean_distance))
        
    fw.close()


if __name__ == '__main__':
    main()
