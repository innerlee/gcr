from typing import List, Optional

import numpy as np
import torch
from packaging import version
from torch import Tensor
from torch.optim.optimizer import Optimizer, required

from mmpretrain.registry import OPTIMIZERS

OPT_ITER = 0
REORTH_STEP = 5
LR_FACTOR = 1.0


@OPTIMIZERS.register_module()
class RSGD(Optimizer):

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 reorth_step=5,
                 lr_factor=1.0,
                 nesterov=False,
                 *,
                 maximize=False,
                 foreach: Optional[bool] = None):
        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening')
        super().__init__(params, defaults)

        self.reorth_step = reorth_step
        global REORTH_STEP
        REORTH_STEP = reorth_step

        global LR_FACTOR
        LR_FACTOR = lr_factor

        assert not nesterov

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad,
                                          momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when
        # value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError(
            'torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        raise NotImplementedError()
    else:
        func = _single_tensor_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize)


def _single_tensor_sgd(params: List[Tensor], d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]], *,
                       weight_decay: float, momentum: float, lr: float,
                       dampening: float, nesterov: bool, maximize: bool,
                       has_sparse_grad: bool):

    global OPT_ITER, REORTH_STEP, LR_FACTOR
    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0 and not hasattr(param, 'geometry'):
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        buf = momentum_buffer_list[i]

        if hasattr(param, 'geometry'):
            if len(np.unique(param.geometry)) == 1:
                subdim = param.geometry[0]
                featdim = param.size(1)
                batch_p = param.reshape(-1, subdim, featdim).permute([0, 2, 1])
                batch_d_p = d_p.reshape(-1, subdim, featdim).permute([0, 2, 1])
                # euclidean grad to riemannian grad
                batch_g = batch_d_p - batch_p @ (
                    param.view(-1, subdim, featdim) @ batch_d_p)

                if version.parse(torch.__version__) < version.parse(
                        '1.13.0') or subdim <= 2:
                    # use cpu for low version of pytorch
                    (batch_U, batch_s, batch_Vt) = torch.linalg.svd(
                        batch_g.cpu(), full_matrices=False)
                    batch_U = batch_U.to(batch_g.device)
                    batch_s = batch_s.to(batch_g.device)
                    batch_Vt = batch_Vt.to(batch_g.device)
                else:
                    # gpu is faster
                    (batch_U, batch_s, batch_Vt) = torch.linalg.svd(
                        batch_g, driver='gesvda', full_matrices=False)

                new_batch_p = (
                    (batch_p @ batch_Vt.permute([0, 2, 1])) * torch.cos(
                        (alpha * LR_FACTOR) * batch_s).reshape(-1, 1, subdim) +
                    batch_U * torch.sin(
                        (alpha * LR_FACTOR) * batch_s).reshape(-1, 1, subdim)
                ) @ batch_Vt
                if OPT_ITER % REORTH_STEP == 0:
                    if subdim <= 2:
                        new_batch_p = torch.linalg.qr(new_batch_p.cpu()).Q.to(
                            new_batch_p.device)
                    else:
                        new_batch_p = torch.linalg.qr(new_batch_p).Q
                OPT_ITER += 1

                param.add_(
                    new_batch_p.permute([0, 2, 1]).reshape(-1, featdim) -
                    param)

            else:
                dims = [0, *np.cumsum(param.geometry)]
                for c, cc in zip(dims[:-1], dims[1:]):
                    p = param[c:cc].T
                    d = d_p[c:cc].T
                    # euclidean grad to riemannian grad
                    g = d - p @ (p.T @ d)
                    (U, s, Vt) = torch.linalg.svd(g, full_matrices=False)
                    new_p = torch.cat((p @ Vt.T, U), 1) @ torch.cat(
                        (torch.diagflat(torch.cos(alpha * s)),
                         torch.diagflat(torch.sin(alpha * s))), 0) @ Vt
                    if OPT_ITER % REORTH_STEP == 0:
                        new_p = torch.linalg.qr(new_p).Q
                    OPT_ITER += 1
                    p.add_(new_p - p)

        else:
            param.add_(d_p, alpha=alpha)
