import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import box_ops


class DETRLoss(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_class_loss(self, indices, class_pred, class_tgt, log=False):
        src_logits = class_pred
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(class_tgt, indices)]
        ).to(src_logits.device)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight.to(src_logits.device),
        )
        losses = {"loss_ce": loss_ce}
        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def get_box_loss(self, indices, box_pred, box_tgt, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = box_pred[idx]
        target_boxes = torch.cat(
            [t[J] for t, (_, J) in zip(box_tgt, indices)], dim=0
        ).to(src_boxes.device)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def forward(self, pred, target):

        indices = self.matcher(pred, target)
        losses = {}
        num_boxes = sum(len(t) for t in target["labels"])
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(pred.values())).device
        )
        class_loss = self.get_class_loss(indices, pred["obj_class"], target["labels"])
        box_loss = self.get_box_loss(indices, pred["bbox"], target["boxes"], num_boxes)

        losses.update(class_loss)
        losses.update(box_loss)

        if "aux_outputs" in pred:
            for i, aux_output in enumerate(pred["aux_outputs"]):
                indices = self.matcher(aux_output, target)
                class_loss = self.get_class_loss(
                    indices, aux_output["obj_class"], target["labels"]
                )
                box_loss = self.get_box_loss(
                    indices, aux_output["bbox"], target["boxes"], num_boxes
                )
                class_loss = {k + f"_{i}": v for k, v in class_loss.items()}
                box_loss = {k + f"_{i}": v for k, v in box_loss.items()}
                losses.update(class_loss)
                losses.update(box_loss)

        return losses
