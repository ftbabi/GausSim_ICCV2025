from mmcv.runner.builder import RUNNERS
from mmcv.runner import IterBasedRunner


@RUNNERS.register_module()
class IterRunner(IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        # Add epoch
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, num_epoch=self.epoch, num_iter=self.iter, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1