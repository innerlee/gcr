from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

from mmpretrain.models import ClsHead
from mmpretrain.registry import MODELS


@MODELS.register_module()
class GrassmannClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 dims,
                 gamma=25.0,
                 orth_init=False,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gamma = gamma
        self.dims = dims
        assert len(dims) == num_classes
        assert len(np.unique(dims)) == 1
        self.subdim = dims[0]
        self.orth_init = orth_init

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, sum(dims), bias=False)
        self.fc.weight.geometry = dims

    def init_weights(self):
        if not self.orth_init:
            return
        with torch.no_grad():
            w = self.fc.weight
            nn.init.normal_(w)
            wt = w.reshape(-1, self.subdim,
                           self.in_channels).permute([0, 2, 1])
            wt = torch.linalg.qr(wt).Q
            w = wt.permute([0, 2, 1]).reshape(-1, self.in_channels)
            self.fc.weight.add_(w - self.fc.weight)
            print('init geom with mean ', self.fc.weight.mean().cpu().item())

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``LinearClsHead``, we just obtain the
        feature of the last stage.
        """
        pre_logit = feats[-1]
        if isinstance(pre_logit, list) and len(pre_logit) == 2:
            # vision transformer
            _, pre_logit = pre_logit
        return pre_logit

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)

        x = normalize(pre_logits, dim=-1)
        cls_score = self.gamma * self.fc(x)

        # The final classification head.
        cls_score = cls_score.reshape(-1, self.num_classes,
                                      self.subdim).norm(dim=-1)
        return cls_score
