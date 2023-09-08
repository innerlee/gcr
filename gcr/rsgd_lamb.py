from mmpretrain.engine import Lamb
from mmpretrain.registry import OPTIMIZERS
from .rsgd import RSGD


@OPTIMIZERS.register_module()
class RLambOptimizer(Lamb):

    def __init__(self,
                 params,
                 lr,
                 *args,
                 reorth_step=1,
                 lr_factor=1.0,
                 momentum=0.9,
                 **kwargs):
        params_rsgd = [
            p for p in params if hasattr(p['params'][0], 'geometry')
        ]
        params_lamb = [
            p for p in params if not hasattr(p['params'][0], 'geometry')
        ]
        self.rsgd = RSGD(
            params_rsgd,
            lr,
            reorth_step=reorth_step,
            lr_factor=lr_factor,
            momentum=momentum)
        super().__init__(params_lamb, lr, *args, **kwargs)

    def zero_grad(self):
        super().zero_grad()
        self.rsgd.zero_grad()

    def step(self):
        # modify lr of rsgd
        lr = self.param_groups[0]['lr']
        for group in self.rsgd.param_groups:
            group['lr'] = lr
        # step
        self.rsgd.step()
        super().step()
