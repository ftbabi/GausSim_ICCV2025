import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from mmgs.core import multi_apply
from kornia.losses import ssim_loss


def ssim_error(pred, label, kernal_size, max_val, weight=None, reduction='sum', avg_factor=None, **kwargs):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = ssim_loss(pred.permute(2,0,1).unsqueeze(0), label.permute(2,0,1).unsqueeze(0), window_size=kernal_size, max_val=max_val, reduction=reduction)
    return loss,


@LOSSES.register_module()
class SSIMLoss(nn.Module):
    """Cross entropy loss

    Args:
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_lovasz'.
    """

    def __init__(self,
                 kernel_size=5,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_ssim',):
        super(SSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = ssim_error
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                ssim_max_val=1.0,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = multi_apply(
            self.criterion,
            cls_score, label,
            kernal_size=self.kernel_size, max_val=ssim_max_val,
            weight=weight, reduction=reduction, avg_factor=avg_factor)[0]
        loss = torch.stack(loss).mean()
        loss *= self.loss_weight
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
