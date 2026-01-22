import numpy as np

from torch._six import inf
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class StepDataAdjusterHook(Hook):

    def __init__(self,
        dataloader,
        init_seq=0,
        step=[0],
        prod=None,
        sum=1.0, **kwargs):
        
        self.dataloader = dataloader
        self.init_seq = init_seq
        self.step = step
        assert not (prod is not None and sum is not None), "Only support one at a time"
        assert not (prod is not None and init_seq == 0), "Productive must has initial step non-zero"
        self.prod = prod
        self.sum = sum

        self.max_seq = init_seq

        super(StepDataAdjusterHook, self).__init__(**kwargs)

    def before_train_epoch(self, runner):
        # Set new maximum sequence length
        progress = runner.epoch

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        
        if self.prod is not None:
            self.max_seq = self.init_seq * (self.prod ** exp)
        elif self.sum is not None:
            self.max_seq = self.init_seq + self.sum * exp
        else:
            raise ValueError(f"The policy input is not correct")
        self._set_random_sequence()
    
    def _set_random_sequence(self):
        length = int(self.max_seq)
        if getattr(self.dataloader.dataset, 'set_sequence', None) is not None:
            self.dataloader.dataset.set_sequence(length)