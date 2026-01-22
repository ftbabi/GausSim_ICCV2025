import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from ..builder import HEADS, build_loss, build_accuracy
from .base_head import BaseHead


@HEADS.register_module()
class SimHead(BaseHead):
    """classification head.

    Args:
        loss (dict | sequnce dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 init_cfg=None,
                 loss_decode=dict(type='L2Loss', reduction='sum', loss_weight=1.0),
                 accuracy=dict(type='L2Accuracy', reduction='mean'),
                 *args,
                 **kwargs):
        super(SimHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        # LOSS
        if isinstance(loss_decode, dict):
            loss_decode = [loss_decode]
        elif isinstance(loss_decode, (list, tuple)):
            pass
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        self.loss_decode = nn.ModuleList()
        for loss in loss_decode:
            self.loss_decode.append(build_loss(loss))
        
        # ACCURACY
        if isinstance(accuracy, dict):
            accuracy = [accuracy]
        elif isinstance(accuracy, (list, tuple)):
            pass
        elif accuracy is None:
            accuracy = []
        else:
            raise TypeError(f'accuracy must be a dict or sequence of dict,\
                but got {type(accuracy)}')
        self.accuracy_decode = nn.ModuleList()
        for accs in accuracy:
            self.accuracy_decode.append(build_accuracy(accs))

    def loss(self, pred, gt_label, reduction_override=None, weight=None, indices=None, indices_weight=None, indices_type=None, term_filter=None, cal_acc=True, **kwargs):
        losses = dict()

        # compute loss
        for loss_decode in self.loss_decode:
            process = term_filter is None
            if term_filter is not None:
                for ln_filter in term_filter:
                    if ln_filter in loss_decode.loss_name:
                        process = True
                        break
            if not process:
                continue
            loss = loss_decode(
                pred,
                gt_label,
                weight=weight,
                reduction_override=reduction_override,
                indices=indices,
                indices_weight=None,
                **kwargs)
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = loss
            else:
                losses[loss_decode.loss_name] += loss

        if cal_acc:
            # compute accuracy
            acc_dict = self.accuracy(pred, gt_label, reduction_override=reduction_override, indices=indices, indices_weight=indices_weight, indices_type=indices_type, term_filter=term_filter, **kwargs)
            losses.update(acc_dict)

        return losses

    def accuracy(self, pred, gt_label, reduction_override=None, indices=None, indices_type=None, term_filter=None, **kwargs):
        acc_dict = defaultdict(list)
        # compute accuracy
        for accuracy_decode in self.accuracy_decode:
            process = term_filter is None
            if term_filter is not None:
                for ln_filter in term_filter:
                    if ln_filter in accuracy_decode.acc_name:
                        process = True
                        break
            if not process:
                continue 
            accs = accuracy_decode(
                pred,
                gt_label,
                indices=indices,
                indices_type=indices_type,
                reduction_override=reduction_override,
                **kwargs)
            for key, val in accs.items():
                acc_dict[key].append(val)
        rst = dict()
        for key, val in acc_dict.items():
            rst[key] = torch.mean(torch.stack(val))
        return rst

    def forward_train_regularize(self, pred, gt_label):
        losses = self.loss(pred, gt_label)
        return losses, pred

    def simple_test(self, cls_score, softmax=True, post_process=True):
        """Inference without augmentation.
        Args:
            cls_score (tuple[Tensor]): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.
        Returns:
            Tensor | list: The inference results.
                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
