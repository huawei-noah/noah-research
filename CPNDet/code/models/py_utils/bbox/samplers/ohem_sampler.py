import torch
import pdb
from ..transforms import bbox2roi
from .base_sampler import BaseSampler


class OHEMSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(OHEMSampler, self).__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals)
        if not hasattr(context, 'num_stages'):
            self.bbox_roi_extractor = context.center_roi_extractor
            self.bbox_head = context.bbox_heads[-1]
        else:
            self.bbox_roi_extractor = context.bbox_roi_extractor[
                context.current_stage]
            self.bbox_head = context.bbox_head[context.current_stage]

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            
            roi_lx = ((2*rois[:, 1] + rois[:, 3])/3).unsqueeze(-1)
            roi_rx = ((rois[:, 1]  + 2*rois[:, 3])/3+1).unsqueeze(-1)
            roi_ty = ((2*rois[:, 2] + rois[:, 4])/3).unsqueeze(-1)
            roi_by = ((rois[:, 2]  + 2*rois[:, 4])/3+1).unsqueeze(-1)

            rois_ = torch.cat((rois[:, 0].unsqueeze(-1), roi_lx, roi_ty, roi_rx, roi_by), dim =1).contiguous()
            bbox_feats = self.bbox_roi_extractor([feats[-1]], rois_)
            cls_score, _ = self.bbox_head(bbox_feats[0])
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduce=False)['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]

    def _sample_pos(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.hard_mining(pos_inds, num_expected, bboxes[pos_inds],
                                    assign_result.labels[pos_inds], feats)

    def _sample_neg(self,
                    assign_result,
                    num_expected,
                    bboxes=None,
                    feats=None,
                    **kwargs):
        # Sample some hard negative samples
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.hard_mining(neg_inds, num_expected, bboxes[neg_inds],
                                    assign_result.labels[neg_inds], feats)
