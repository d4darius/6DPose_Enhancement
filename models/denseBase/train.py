# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataload.dataloader import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet, GNNPoseNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import wandb
from torch_geometric.data import Batch

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
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 32, help='batch size')
parser.add_argument('--workers', type=int, default = 4, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
parser.add_argument('--gnn', action='store_true', default=False, help='start training on the geometric model')
opt = parser.parse_args()

# Initialize W&B
wandb.init(
    project="6D-Pose-Estimation",  # Replace with your project name
    config={
        "dataset": opt.dataset,
        "batch_size": opt.batch_size,
        "learning_rate": opt.lr,
        "epochs": opt.nepoch,
        "refinement_iterations": opt.iteration,
    }
)

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    #--------------------------------------------------------
    # DATASET INITIALIZATION: Setup the dataset
    #--------------------------------------------------------
    if opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'checkpoints/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 5
    else:
        print('Unknown dataset')
        return
    
    #--------------------------------------------------------
    # MODEL INITIALIZATION: Setup the estimator and refiner models
    #--------------------------------------------------------
    if opt.gnn:
        print("Using GNN DenseFusion")
        estimator = GNNPoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.to(device)

        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    else:  
        print("Using Simple DenseFusion")  
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.to(device)
        refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
        refiner.to(device)

        if opt.resume_posenet != '':
            estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

        if opt.resume_refinenet != '':
            refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
            opt.refine_start = True
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
        else:
            opt.refine_start = False
            opt.decay_start = False
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    #--------------------------------------------------------
    # DATASET LOADING: Setup the dataloader and dataset
    #--------------------------------------------------------
    if opt.dataset == 'linemod':
        dataset = PoseDataset_linemod(opt.dataset_root, 'train', num_points=opt.num_points, add_noise=True, refine=opt.refine_start, device=device, sampling='FPS')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True, collate_fn=dataset.center_pad_collate)
    if opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod(opt.dataset_root, 'test', num_points=opt.num_points, add_noise=False, noise_trans=0.0, refine=opt.refine_start, device=device, sampling='FPS')
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True, collate_fn=dataset.center_pad_collate)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

#--------------------------------------------------------
# EPOCH LOOP: Loop over the epochs repeating training and following with testing
#--------------------------------------------------------
    for epoch in range(opt.start_epoch, opt.nepoch):
        #--------------------------------------------------------
        # TRAINING PART: Train the model
        #--------------------------------------------------------
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                if 'error' in data:
                    print(data['error'])
                    continue
                points = data['cloud']
                choose = data['choose']
                img = data['image']
                target = data['target']
                model_points = data['model_points']
                graph_batch = Batch.from_data_list(data['graph'])
                idx = data['obj_id']
                points, choose, img, target, model_points, graph_batch, idx = points.to(device), \
                                                                              choose.to(device), \
                                                                              img.to(device), \
                                                                              target.to(device), \
                                                                              model_points.to(device), \
                                                                              graph_batch.to(device), \
                                                                              idx.to(device)
                if opt.gnn:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, graph_batch, idx)
                else:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                # Log metrics to W&B
                wandb.log({
                    "epoch": epoch,
                    "batch": train_count,
                    "loss": loss.item(),
                    "distance": dis.item(),
                })
                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                #if train_count % opt.batch_size == 0:
                logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_count, int(train_count*idx.size()[0]), train_dis_avg))
                optimizer.step()
                optimizer.zero_grad()
                train_dis_avg = 0

                if train_count != 0 and train_count % 100 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        if opt.gnn:
                            torch.save(estimator.state_dict(), '{0}/gnn_pose_model_current.pth'.format(opt.outf))
                        else:
                            torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))
                logger.info('Finish train time {0}'.format(time.strftime("%Hh %Mm %Ss")))

        if opt.refine_start:
            torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
        else:
            torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))
                        
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        #--------------------------------------------------------
        # TESTING PART: Test the model
        #--------------------------------------------------------
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        #refiner.eval()

        with torch.no_grad():
            for j, data in enumerate(testdataloader, 0):
                if 'error' in data:
                    print(data['error'])
                    continue
                points = data['cloud']
                choose = data['choose']
                img = data['image']
                target = data['target']
                model_points = data['model_points']
                graph_batch = Batch.from_data_list(data['graph'])
                idx = data['obj_id']
                points, choose, img, target, model_points, graph_batch, idx = points.to(device), \
                                                                              choose.to(device), \
                                                                              img.to(device), \
                                                                              target.to(device), \
                                                                              model_points.to(device), \
                                                                              graph_batch.to(device), \
                                                                              idx.to(device)
                if opt.gnn:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, graph_batch, idx)
                else:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

                test_dis += dis.item()
                # Log metrics to W&B
                wandb.log({
                    "epoch": epoch,
                    "test_batch": test_count,
                    "test_distance": dis.item(),
                })
                logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), int(test_count*idx.size()[0]), dis))

                test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                if opt.gnn:
                    torch.save(estimator.state_dict(), '{0}/gnn_pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
                else:
                    torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
        
        #--------------------------------------------------------
        # LR ADJUSTMENT PART: Adjust the learning rate and weights
        #--------------------------------------------------------
        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        #--------------------------------------------------------
        # TRIGGER REFINEMENT PART: Trigger the refinement training
        #--------------------------------------------------------
        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'linemod':
                dataset = PoseDataset_linemod(opt.dataset_root, 'train', num_points=opt.num_points, add_noise=True, refine=opt.refine_start, device=device)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True, collate_fn=dataset.center_pad_collate)
            if opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod(opt.dataset_root, 'test', num_points=opt.num_points, add_noise=False, noise_trans=0.0, refine=opt.refine_start, device=device)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True, collate_fn=dataset.center_pad_collate)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()
