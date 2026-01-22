# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ACCURACY
# from mmcv.ops import QueryAndGroup
# from mmgs.core import multi_apply


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.
    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.
    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res

def accuracy_l2(pred, label, prefix='', merge=True, **kwargs):
    '''
        pred: list[n_cam, 1*image]
        label: list[n_cam, 1*image]
    '''
    bs = len(pred)

    # import plotly.express as px
    # for i in range(bs):
    #     pred_img = pred[i] * 255
    #     gt_img = label[i] * 255
    #     pred_img = pred_img.detach().contiguous().cpu().numpy().astype(np.uint8)
    #     gt_img = gt_img.detach().contiguous().cpu().numpy().astype(np.uint8)
    #     fig = px.imshow(pred_img)
    #     fig.show()
    #     fig = px.imshow(gt_img)
    #     fig.show()

    acc_dict = defaultdict(list)
    # element-wise losses
    l2_error = [
        torch.sqrt(torch.sum(
            F.mse_loss(pred[i], label[i], reduction='none'), 
            dim=-1)) 
        for i in range(bs)]

    # Calculate per outfit error
    for cam_id in range(bs):
        acc_dict[cam_id].append(l2_error[cam_id])
    
    # Avg batch here. During test the bs == 1
    acc_dict = {
        f"{prefix}.{key}": torch.mean(torch.stack(val))
        for key, val in acc_dict.items()
    }

    if not merge:
        return acc_dict

    acc_dict = {
        f"{prefix}.acc": torch.mean(torch.stack(list(acc_dict.values())))
    }
    return acc_dict

@ACCURACY.register_module()
class L2Accuracy(nn.Module):

    def __init__(self,
                 reduction='mean',
                 acc_name='accuracy_l2'):
        """Module to calculate the accuracy.
        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super(L2Accuracy, self).__init__()
        self.reduction = reduction
        self._acc_name = acc_name

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy_l2(pred, target, prefix=self.acc_name, **kwargs)
    
    @property
    def acc_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._acc_name