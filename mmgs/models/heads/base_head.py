# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head."""

    def __init__(self, init_cfg=None, *args, **kwargs):
        super(BaseHead, self).__init__(init_cfg)

    # @auto_fp16()
    # @abstractmethod
    # def forward(self, inputs):
    #     """Placeholder of forward function."""
    #     pass

    @abstractmethod
    def forward_train_regularize(self, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, **kwargs):
        pass