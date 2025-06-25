# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import argparse
import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataload.dataloader import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, GNNPoseNet
from lib.loss import Loss
from lib.utils import setup_logger
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
parser.add_argument('--dataset', type=str, default = 'linemod', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 32, help='batch size')
parser.add_argument('--workers', type=int, default = 8, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
parser.add_argument('--gnn', action='store_true', default=False, help='start training on the geometric model')
parser.add_argument('--feat', type=str, default = 'color',  help='selector for the feature to be used in GIN')
opt = parser.parse_args()

# Initialize W&B
wandb.init(
    project="6D-Pose-Estimation",  # Replace with your project name
    config={
        "dataset": opt.dataset,
        "batch_size": opt.batch_size,
        "learning_rate": opt.lr,
        "epochs": opt.nepoch
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
    # MODEL INITIALIZATION: Setup the estimator
    #--------------------------------------------------------
    if opt.gnn:
        print("Using GNN DenseFusion")
        estimator = GNNPoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.to(device)
    else:  
        print("Using Simple DenseFusion")  
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        estimator.to(device)

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
    
    opt.decay_start = False
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    #--------------------------------------------------------
    # DATASET LOADING: Setup the dataloader and dataset
    #--------------------------------------------------------
    if opt.dataset == 'linemod':
        dataset = PoseDataset_linemod(opt.dataset_root, 'train', num_points=opt.num_points, add_noise=True, device=device, sampling='random')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True, collate_fn=dataset.center_pad_collate)
    if opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod(opt.dataset_root, 'test', num_points=opt.num_points, add_noise=False, noise_trans=0.0, device=device, sampling='random')
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True, collate_fn=dataset.center_pad_collate)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))
    print(f'Using {opt.feat}')

    criterion = Loss(opt.num_points_mesh, opt.sym_list)

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
        train_frame = 0
        train_dis_avg = 0.0
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
                graph_batch = data['graph']
                idx = data['obj_id']
                points, choose, img, target, model_points, graph_batch, idx = points.to(device), \
                                                                              choose.to(device), \
                                                                              img.to(device), \
                                                                              target.to(device), \
                                                                              model_points.to(device), \
                                                                              graph_batch.to(device), \
                                                                              idx.to(device)
                if opt.gnn:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, graph_batch, idx, opt.feat)
                else:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w)

                # Log metrics to W&B
                wandb.log({
                    "epoch": epoch,
                    "batch": train_count,
                    "loss": loss.item(),
                    "distance": dis.item(),
                })
                
                loss.backward()

                train_dis_avg += dis.item()
                train_count += 1
                train_frame += idx.size()[0]

                logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4} {5}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, train_count, train_frame, train_dis_avg, ''))
                optimizer.step()
                optimizer.zero_grad()
                train_dis_avg = 0

            if train_count != 0:
                if opt.gnn:
                    torch.save(estimator.state_dict(), '{0}/gnn_pose_model_current.pth'.format(opt.outf))
                else:
                    torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

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
                graph_batch = data['graph']
                idx = data['obj_id']
                points, choose, img, target, model_points, graph_batch, idx = points.to(device), \
                                                                              choose.to(device), \
                                                                              img.to(device), \
                                                                              target.to(device), \
                                                                              model_points.to(device), \
                                                                              graph_batch.to(device), \
                                                                              idx.to(device)
                if opt.gnn:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, graph_batch, idx, opt.feat)
                else:
                    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w)

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
            if opt.gnn:
                torch.save(estimator.state_dict(), '{0}/gnn_pose_model_{1}_{2}_{3}.pth'.format(opt.outf, epoch, test_dis, opt.feat))
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

if __name__ == '__main__':
    main()
