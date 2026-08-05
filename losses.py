import torch.nn as nn


class IoULoss(nn.Module):
    """Soft intersection-over-union loss used in Eq. (12)."""

    def forward(self, preds, gt_masks):
        if isinstance(preds, (list, tuple)):
            return sum(self._single_loss(pred, gt_masks) for pred in preds) / len(preds)
        return self._single_loss(preds, gt_masks)

    @staticmethod
    def _single_loss(pred, gt_masks):
        smooth = 1.0
        intersection = (pred * gt_masks).sum()
        union = pred.sum() + gt_masks.sum() - intersection
        return 1.0 - (intersection + smooth) / (union + smooth)
