import torch
import pdb
from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class CenterRegionAssigner(BaseAssigner):
    def __init__(self, region_size = 3, score_thr=0.1):
        self.region_size = region_size

    def assign(self, bboxes, gt_bboxes, gt_labels=None):
#         ########################################################
#         gt_bboxes = gt_bboxes[:5,:]
#         gt_labels = gt_labels[:5]
#         ########################################################
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
         
        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        assigned_gt_inds   = bboxes.new_full((num_bboxes, ), -1, dtype=torch.long)
        
        center_x, center_y = (gt_bboxes[:, 2] + gt_bboxes[:, 0])/2., (gt_bboxes[:, 3] + gt_bboxes[:, 1])/2.  
        
        centers = torch.cat((center_x.unsqueeze(-1), center_y.unsqueeze(-1), gt_labels.float().unsqueeze(-1)), dim = 1)
        
        center_matrix = centers.permute(1,0).unsqueeze(-1).expand(3, centers.size(0), bboxes.size(0)).contiguous()  
        bbox_matrix  = bboxes.permute(1,0).unsqueeze(1).expand(4, centers.size(0), bboxes.size(0)).contiguous() 
        
        bbox_lx = (2*bbox_matrix[0, ...] + bbox_matrix[2, ...])/3
        bbox_rx = (bbox_matrix[0, ...]  + 2*bbox_matrix[2, ...])/3+1
        bbox_ty = (2*bbox_matrix[1, ...] + bbox_matrix[3, ...])/3
        bbox_by = (bbox_matrix[1, ...]  + 2*bbox_matrix[3, ...])/3+1
        
        ind_lx = (center_matrix[0, ...] - bbox_lx) > 0
        ind_rx = (center_matrix[0, ...] - bbox_rx) < 0
        ind_ty = (center_matrix[1, ...] - bbox_ty) > 0
        ind_by = (center_matrix[1, ...] - bbox_by) < 0
        
        inds_matrix = (ind_lx & ind_rx & ind_ty & ind_by).float()
        neg_inds    = inds_matrix.sum(0) == 0
        pos_inds    = inds_matrix.sum(0) > 0
        assigned_gt_inds[neg_inds] = 0
        
        pos_index = (inds_matrix + overlaps).max(dim=0)[1]
        assigned_gt_inds[pos_inds] = pos_index[pos_inds] + 1
        
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
