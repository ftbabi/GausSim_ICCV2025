from torch._six import inf
from mmcv.runner.hooks import HOOKS, Hook, OptimizerHook


@HOOKS.register_module()
class CumulativeHook(OptimizerHook):

    def __init__(self, iter, scaler=1.0, grad_clip=None, **kwargs):
        super(CumulativeHook, self).__init__(grad_clip)
        assert iter >= 2
        self.iter_bound = iter
        self.loss_sum = 0
        self.scaler = scaler

    def after_train_iter(self, runner):
        cur_iter = runner.iter
        self.loss_sum += runner.outputs['loss']
        if (cur_iter+1) % self.iter_bound == 0:
            runner.optimizer.zero_grad()
            self.loss_sum = (self.loss_sum * self.scaler)/self.iter_bound
            self.loss_sum.backward()
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
            runner.optimizer.step()
            self.loss_sum = 0
    