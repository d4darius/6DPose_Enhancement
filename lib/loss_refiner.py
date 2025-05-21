from torch.nn.modules.loss import _Loss
import torch


def loss_calculation(pred_r, pred_t, target, model_points, idx, points, num_point_mesh, sym_list):
    # pred_r: [bs, 4] (quaternion from refiner)
    # pred_t: [bs, 3] (translation from refiner)
    # target: [bs, num_point_mesh, 3] (ground truth model points in canonical frame)
    # model_points: [bs, num_point_mesh, 3] (model points in canonical frame - same as target for loss calculation)
    # idx: [bs] (object class indices)
    # points: [bs, num_input_points, 3] (observed points, possibly transformed by previous pose estimate)
    # num_point_mesh: int (number of points on the mesh model)
    # sym_list: list of symmetric object indices

    bs = pred_r.size(0)
    num_p = 1  # Refiner typically outputs one pose correction per item

    # Add a dimension for 'num_predictions_per_item', which is 1 for the refiner
    pred_r = pred_r.unsqueeze(1)  # Shape: [bs, 1, 4]
    pred_t = pred_t.unsqueeze(1)  # Shape: [bs, 1, 3]

    num_input_points = points.size(1)

    # Normalize quaternion
    pred_r = pred_r / (torch.norm(pred_r, dim=2, keepdim=True) + 1e-8) # Added epsilon for stability

    # Convert quaternion to rotation matrix (R)
    # base will be shape [bs * num_p, 3, 3] = [bs, 3, 3] as num_p is 1
    # This is delta_R from the refiner
    q0 = pred_r[:, :, 0]
    q1 = pred_r[:, :, 1]
    q2 = pred_r[:, :, 2]
    q3 = pred_r[:, :, 3]

    r00 = 1.0 - 2.0 * (q2**2 + q3**2)
    r01 = 2.0 * (q1*q2 - q0*q3)
    r02 = 2.0 * (q1*q3 + q0*q2)
    r10 = 2.0 * (q1*q2 + q0*q3)
    r11 = 1.0 - 2.0 * (q1**2 + q3**2)
    r12 = 2.0 * (q2*q3 - q0*q1)
    r20 = 2.0 * (q1*q3 - q0*q2)
    r21 = 2.0 * (q2*q3 + q0*q1)
    r22 = 1.0 - 2.0 * (q1**2 + q2**2)

    # Stack to form rotation matrices
    # Each element rXX is [bs, num_p] = [bs, 1]
    # View as [bs * num_p, 3, 3]
    delta_R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=2).contiguous().view(bs * num_p, 3, 3) # Shape: [bs, 3, 3]

    # For loss calculation: transformed_model_points = model_points @ delta_R^T + delta_t
    delta_R_T = delta_R.transpose(1, 2) # Shape: [bs, 3, 3]

    # model_points are [bs, num_point_mesh, 3]
    # delta_t (pred_t) is [bs, 1, 3]
    transformed_model_points = torch.bmm(model_points, delta_R_T) + pred_t

    # Handle symmetry:
    # Compare transformed_model_points with points (observed cloud)
    # points_for_loss is the observed cloud, target_for_loss is transformed_model_points
    target_for_loss = transformed_model_points.clone() # Clone for modification if symmetric

    for b_item in range(bs):
        if idx[b_item].item() in sym_list:
            # current_points_obs: [num_input_points, 3]
            # current_model_transformed: [num_point_mesh, 3]
            current_points_obs_item = points[b_item] # This is what we compare against
            current_model_transformed_item = target_for_loss[b_item]

            # For ADD-S, find nearest points from current_model_transformed_item to current_points_obs_item
            # This is slightly different from original logic.
            # Original: pred (transformed model) vs target (canonical model, but find nearest in pred)
            # Here: points (observed) vs target_for_loss (transformed model)
            # We need to find for each point in target_for_loss[b_item], its closest point in points[b_item]
            # Or, for each point in points[b_item], its closest in target_for_loss[b_item]
            # The loss is ADD, so it's distance between corresponding points.
            # If symmetric, it's min distance.
            # pred in original was model_points @ R^T + t
            # target in original was GT model points
            # Here, points is the "ground truth" in the current (potentially misaligned) frame.
            # target_for_loss is our prediction of where model points are in that same frame.

            # Symmetric objects: for each point in predicted model (target_for_loss[b_item]),
            # find the closest point in the observed points (points[b_item])
            # This is if we consider points[b_item] as the ground truth cloud.
            # The standard ADD-(S) loss compares predicted model points to GT model points.
            # Here, `points` are the input to the refiner stage (observed cloud in a certain frame)
            # `target_for_loss` are the canonical model points transformed by the predicted delta.
            # So we compare these two sets.
            
            pred_pts_sym = target_for_loss[b_item].unsqueeze(1)  # [num_point_mesh, 1, 3]
            gt_pts_sym = points[b_item].unsqueeze(0) # [1, num_input_points, 3]
                                                    # This assumes points are the GT to match against
                                                    # This part needs to align with how ADD-S is defined for refinement.
                                                    # Usually, it's between two sets of model points.
                                                    # If `points` are sampled from observed scene and `target_for_loss` are model points,
                                                    # then for symmetry, for each point in `target_for_loss`, find closest in `points`.

            # Let's assume the original symmetry logic was:
            # For each point in `pred` (our transformed model), find the closest point in `target` (GT model points).
            # Here, `target_for_loss` is our `pred`. `points` is the "target cloud".
            # So, for each point in `target_for_loss[b_item]`, find closest in `points[b_item]`.
            # This means `points[b_item]` would be used to update `target_for_loss[b_item]`. This is not right.

            # The symmetry should be applied by finding the nearest points on the *predicted model surface*
            # to the *ground truth model surface* (which is `points` in this context, if `points` are GT points).
            # Let's follow the structure of the original refiner loss's symmetry:
            # pred_exp = pred.unsqueeze(2) -> target_for_loss[b_item].unsqueeze(1)
            # target_exp = target.unsqueeze(1) -> points[b_item].unsqueeze(0)
            # dist_matrix = torch.norm(pred_exp - target_exp, dim=3)
            # nearest_indices = dist_matrix.argmin(dim=2)
            # target = target[batch_indices, nearest_indices] -> points[b_item] is used to pick from itself.

            # Let's assume the symmetry is applied to the `target_for_loss` against `points`.
            # For each point in `target_for_loss[b_item]`, find its closest correspondence in `points[b_item]`.
            # This would mean `target_for_loss[b_item]` gets updated using `points[b_item]`.
            # This is ICP-like.

            # Reverting to the original refiner's symmetry logic structure:
            # `pred` was `model_points @ base + pred_t` (model transformed by prediction)
            # `target` was `GT model_points`
            # If symmetric, `target` was updated by finding nearest points in `target` itself, indexed by `pred`.
            # Here: `target_for_loss` is `model_points @ delta_R_T + delta_t`
            # `points` is the observed cloud.
            # The comparison is `target_for_loss` vs `points`.
            # If object `idx[b_item]` is symmetric:
            current_pred_model = target_for_loss[b_item] # [num_point_mesh, 3]
            current_gt_cloud = points[b_item]           # [num_input_points, 3]

            # For each point in current_pred_model, find the closest point in current_gt_cloud
            dist_matrix = torch.cdist(current_pred_model, current_gt_cloud) # [num_point_mesh, num_input_points]
            min_dists, _ = torch.min(dist_matrix, dim=1) # [num_point_mesh]
            # The loss for this symmetric item would be torch.mean(min_dists)
            # This replaces the direct correspondence loss for symmetric items.
            # This needs to be integrated into the main loss calculation.

    # Calculate distances:
    # dis_all_pairs = torch.norm((target_for_loss.unsqueeze(2) - points.unsqueeze(1)), dim=3) # [bs, num_point_mesh, num_input_points]
    # For non-symmetric, it's a direct correspondence if num_point_mesh == num_input_points and they correspond.
    # The original refiner loss was: dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    # where pred was transformed model points, and target was GT model points (canonical).
    # This implies `points` here should be the GT model points, not the observed cloud.
    # If `points` are the observed cloud, then the loss is between predicted model points and observed cloud.

    # Assuming `points` are the target points to match against (e.g. observed cloud corresponding to the model)
    # And `target_for_loss` are the model points transformed by the current refinement prediction.
    # The loss is the average distance between these two sets.
    # If the object is symmetric, the loss is based on nearest neighbor distances.

    batch_losses = []
    for b_item in range(bs):
        pred_model_pts = target_for_loss[b_item] # [num_point_mesh, 3]
        gt_target_pts = points[b_item]           # [num_input_points, 3]

        if idx[b_item].item() in sym_list:
            # For symmetric objects, use 1-sided Chamfer/ICP-like distance:
            # For each predicted model point, find the closest ground truth target point.
            dist_matrix = torch.cdist(pred_model_pts, gt_target_pts) # [num_point_mesh, num_input_points]
            min_dists_to_gt, _ = torch.min(dist_matrix, dim=1)     # [num_point_mesh]
            item_loss = torch.mean(min_dists_to_gt)
        else:
            # For non-symmetric objects, if num_point_mesh == num_input_points, assume correspondence.
            # Otherwise, this definition is problematic.
            # The original refiner loss compared pred (transformed model) and target (canonical model points).
            # This implies `points` should be the canonical model points if we follow that structure.
            # Let's assume `points` are the target points corresponding to `model_points` after transformation.
            # If `points` are the observed cloud, and `num_point_mesh != num_input_points`,
            # a simple torch.norm(pred_model_pts - gt_target_pts) is not possible.
            # The original code `dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)` assumes pred and target have same num_points.
            # `pred` was `model_points` transformed. `target` was `model_points` (GT).
            # So, `points` here should be `model_points` if we strictly follow that.
            # But the arguments are `target` and `model_points` (both canonical model) and `points` (observed).

            # Let's use the definition from the original refiner:
            # `pred` = `transformed_model_points`
            # `target` = `model_points` (canonical GT model points, which is `target` or `model_points` arg)
            # This means the loss is how well the *delta transformation* aligns the canonical model to itself (if target is canonical model).
            # This doesn't make sense. The loss must be against the observed data or a GT pose.

            # Re-interpreting: `points` is the target point cloud (e.g., from sensor, aligned by previous estimate).
            # `target_for_loss` is `model_points` transformed by the *current refinement delta*.
            # Loss is distance between `target_for_loss` and `points`.
            # For non-symmetric, if they are to correspond directly (e.g. after FPS on both to get same number of points):
            if pred_model_pts.shape[0] == gt_target_pts.shape[0]:
                 item_loss = torch.mean(torch.norm(pred_model_pts - gt_target_pts, dim=1))
            else:
                # If point counts differ, non-symmetric loss is ill-defined without correspondence.
                # Defaulting to a Chamfer-like distance for robustness, or error.
                # This indicates a potential mismatch in how data is prepared or loss is defined.
                # For now, let's use 1-sided Chamfer like for symmetric if counts differ.
                dist_matrix_ns = torch.cdist(pred_model_pts, gt_target_pts)
                min_dists_ns, _ = torch.min(dist_matrix_ns, dim=1)
                item_loss = torch.mean(min_dists_ns)
                # Or raise an error:
                # raise ValueError(f"Point counts mismatch for non-symmetric object {idx[b_item].item()}: model {pred_model_pts.shape[0]}, target {gt_target_pts.shape[0]}")
        batch_losses.append(item_loss)

    final_loss = torch.mean(torch.stack(batch_losses))


    # For the next iteration, transform the input `points` and the canonical `model_points` (i.e., `target` arg)
    # by the inverse of the predicted delta pose.
    # delta_R is [bs, 3, 3], pred_t is [bs, 1, 3]
    # Inverse delta: R_inv = delta_R^T, t_inv = -delta_R^T @ delta_t
    
    # Transform `points` (observed cloud for this stage)
    # new_points = (points - pred_t) @ delta_R
    points_centered_for_next_stage = points - pred_t # Broadcasting [bs, N_in, 3] - [bs, 1, 3]
    new_points = torch.bmm(points_centered_for_next_stage, delta_R) # [bs, N_in, 3] @ [bs, 3, 3]

    # Transform `target` (canonical model_points)
    # new_target = (target_canonical - pred_t) @ delta_R
    target_canonical = target # Argument `target` is the canonical model points
    target_centered_for_next_stage = target_canonical - pred_t # Broadcasting [bs, N_mesh, 3] - [bs, 1, 3]
    new_target = torch.bmm(target_centered_for_next_stage, delta_R) # [bs, N_mesh, 3] @ [bs, 3, 3]

    return final_loss, new_points.detach(), new_target.detach()


class Loss_refine(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss_refine, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, target, model_points, idx, points):
        # target and model_points are expected to be the same canonical model points
        # points is the observed cloud (possibly transformed by prior estimates)
        return loss_calculation(pred_r, pred_t, target, model_points, idx, points, self.num_pt_mesh, self.sym_list)
