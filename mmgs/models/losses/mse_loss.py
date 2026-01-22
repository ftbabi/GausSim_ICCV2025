import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from mmgs.core import multi_apply


def mean_squared_error(pred, label, vert_mask=None, weight=None, reduction='mean', avg_factor=None):
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
    loss = F.mse_loss(pred, label, reduction='none')
    if vert_mask is not None and vert_mask.shape[0] == loss.shape[0]:
        loss = loss * vert_mask

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    assert not torch.isnan(loss)

    return loss,


@LOSSES.register_module()
class MSELoss(nn.Module):
    """Cross entropy loss

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='loss_inertia',):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = mean_squared_error
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                vert_mask=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        bs = len(cls_score)
        loss = multi_apply(
            self.criterion,
            cls_score, label,
            vert_mask if vert_mask is not None else [None]*bs,
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