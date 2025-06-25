from torch.nn.modules.loss import _Loss
import torch


def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, num_point_mesh, sym_list):
    bs, num_p, _ = pred_t.size()
    
    # Normalize quaternions
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    
    # Convert quaternions to rotation matrices - keeping batch structure
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                     (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                     (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                     (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                     (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                     (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                     (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                     (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                     (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()
    
    # Reshape tensors for batch matrix multiplication
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target.clone()  # Use clone to avoid unintended modifications
    
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t.clone()
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    # Apply rotation and translation
    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    # Create mask of symmetric objects
    sym_mask = torch.tensor([i.item() in sym_list for i in idx], device=pred.device)
    
    if sym_mask.any():
        # Process each symmetric object separately
        for b in range(bs):
            if sym_mask[b]:
                # Get indices for this batch item
                start_idx = b * num_p
                end_idx = (b + 1) * num_p
                
                # Extract just this object's predictions and targets
                obj_pred = pred[start_idx:end_idx]  # Shape: [num_p, num_point_mesh, 3]
                obj_target = target[start_idx:end_idx]  # Shape: [num_p, num_point_mesh, 3]
                
                # Process this object's symmetry (much smaller tensors)
                obj_pred_exp = obj_pred.unsqueeze(2)  # Shape: [num_p, num_point_mesh, 1, 3]
                obj_target_exp = obj_target.unsqueeze(1)  # Shape: [num_p, 1, num_point_mesh, 3]
                
                # Calculate distances using much smaller matrices
                obj_dist_matrix = torch.norm(obj_pred_exp - obj_target_exp, dim=3)
                obj_nearest_indices = obj_dist_matrix.argmin(dim=2)
                
                # Get indices for gather operation
                obj_batch_indices = torch.arange(num_p, device=pred.device).unsqueeze(1).repeat(1, obj_pred.size(1))
                
                # Update only this object's targets
                obj_target_updated = obj_target[obj_batch_indices, obj_nearest_indices]
                target[start_idx:end_idx] = obj_target_updated
    
    # Compute distances and loss
    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    # Find highest confidence prediction per batch item
    pred_c = pred_c.view(bs, num_p)
    which_max = pred_c.argmax(dim=1)
    dis = dis.view(bs, num_p)
    
    # Create batch-wise transformations
    t_list = []
    new_points_list = []
    new_target_list = []
    
    for b in range(bs):
        # Get best prediction for this batch item
        b_which = which_max[b]
        flat_idx = b * num_p + b_which
        
        # Get translation
        b_t = (ori_t[flat_idx] + points[flat_idx]).squeeze(0)
        
        # Get rotation
        b_ori_base = ori_base[flat_idx]
        
        # Store transformation
        t_list.append(b_t)
        
        # Transform points
        b_points = points.view(bs * num_p, 3)[b*num_p:(b+1)*num_p]
        b_points_centered = b_points - b_t
        b_new_points = torch.matmul(b_points_centered, b_ori_base)
        new_points_list.append(b_new_points)
        
        # Transform target
        b_target = ori_target[b*num_p, :, :]
        b_target_centered = b_target - b_t
        b_new_target = torch.matmul(b_target_centered, b_ori_base)
        new_target_list.append(b_new_target)
    
    # Stack batch results
    new_points = torch.cat(new_points_list, dim=0).view(bs, num_p, 3)
    new_target = torch.stack(new_target_list)
    
    # Get per-batch metrics for return
    batch_dis = torch.mean(torch.stack([dis[b, which_max[b]] for b in range(bs)]))
      
    return loss, batch_dis, new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, self.num_pt_mesh, self.sym_list)
