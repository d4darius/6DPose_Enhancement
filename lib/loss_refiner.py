from torch.nn.modules.loss import _Loss
import torch


def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    bs = pred_r.size(0)
    num_p = 1
    num_input_points = points.size(1)

    # Ensure proper shape
    pred_r = pred_r.view(bs, num_p, -1)
    pred_t = pred_t.view(bs, num_p, -1)
    
    # Normalize quaternions
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
    
    # Convert quaternions to rotation matrices
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
    
    ori_base = base.clone()
    base = base.contiguous().transpose(2, 1).contiguous()
    
    # Reshape tensors for batch matrix multiplication
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target.clone()
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t.clone()

    # Apply rotation and translation
    pred = torch.add(torch.bmm(model_points, base), pred_t)

    # Handle symmetry for all batch items
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
                
                # Process this object's symmetry
                obj_pred_exp = obj_pred.unsqueeze(2)  # Shape: [num_p, num_point_mesh, 1, 3]
                obj_target_exp = obj_target.unsqueeze(1)  # Shape: [num_p, 1, num_point_mesh, 3]
                
                # Calculate distances
                obj_dist_matrix = torch.norm(obj_pred_exp - obj_target_exp, dim=3)
                obj_nearest_indices = obj_dist_matrix.argmin(dim=2)
                
                # Get indices for gather operation
                obj_batch_indices = torch.arange(num_p, device=pred.device).unsqueeze(1).repeat(1, obj_pred.size(1))
                
                # Update only this object's targets
                obj_target_updated = obj_target[obj_batch_indices, obj_nearest_indices]
                target[start_idx:end_idx] = obj_target_updated

    # Compute distances
    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1).view(bs, num_p)
    
    # Process each batch item separately for refinement
    new_points_list = []
    new_target_list = []
    
    for b in range(bs):
        # Get transformation for this batch item
        b_flat_idx = b * num_p
        
        # Get translation and rotation for this batch
        b_t = ori_t[b_flat_idx]
        b_ori_base = ori_base[b_flat_idx]
        
        # Transform points for this batch
        b_points = points[b].unsqueeze(0)  # [1, num_input_points, 3]
        b_points_centered = b_points - b_t
        b_new_points = torch.bmm(b_points_centered, b_ori_base)
        new_points_list.append(b_new_points)
        
        # Transform target for this batch
        b_target = ori_target[b_flat_idx].unsqueeze(0)  # [1, num_point_mesh, 3]
        b_target_centered = b_target - b_t
        b_new_target = torch.bmm(b_target_centered, b_ori_base)
        new_target_list.append(b_new_target)
    
    # Stack batch results
    new_points = torch.cat(new_points_list, dim=0)
    new_target = torch.cat(new_target_list, dim=0)
    
    # Get per-batch metrics
    batch_dis = dis.squeeze(1)  # [bs]
    
    return batch_dis, new_points.detach(), new_target.detach()


class Loss_refine(_Loss):
    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)
    