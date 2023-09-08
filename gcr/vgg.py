import torch.nn as nn

from mmpretrain.models import VGG
from mmpretrain.registry import MODELS


@MODELS.register_module()
class VGG_(VGG):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
