import torch
import torch.nn as nn
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        
        bs, num_queries = outputs["obj_class"].shape[:2]
        
        out_prob = outputs["obj_class"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["bbox"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_ids = torch.cat([v for v in targets["labels"]]).to(out_prob.device)
        tgt_bbox = torch.cat([v for v in targets["boxes"]]).to(out_bbox.device)
        
        cost_class = -out_prob[:, tgt_ids]
        
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v) for v in targets["boxes"]]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]