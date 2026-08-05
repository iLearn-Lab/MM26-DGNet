import torch.nn as nn

from losses import IoULoss
from model.dgnet import DGNet


class DGNetModel(nn.Module):
    """DGNet and its pixel-level IoU loss."""

    def __init__(self, model_name="DGNet"):
        super().__init__()
        if model_name != "DGNet":
            raise ValueError(
                f"Unsupported model '{model_name}'. This release contains "
                "only DGNet."
            )
        self.model_name = model_name
        self.iou_loss = IoULoss()
        self.model = DGNet()

    def forward(self, data, text_eot_targets, text_eot_bg):
        return self.model(data, text_eot_targets, text_eot_bg)

    def loss(self, pred, gt_mask):
        return self.iou_loss(pred, gt_mask)
