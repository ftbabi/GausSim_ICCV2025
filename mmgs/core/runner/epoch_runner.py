from mmcv.runner.builder import RUNNERS
from mmcv.runner import EpochBasedRunner

from typing import Any


@RUNNERS.register_module()
class EpochRunner(EpochBasedRunner):

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        # Input the number of epoch for auto adjustment
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, num_epoch=self.epoch, num_iter=self.iter,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, num_epoch=self.epoch, num_iter=self.iter, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs