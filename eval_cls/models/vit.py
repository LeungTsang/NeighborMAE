# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from util.pos_embed import get_2d_sincos_pos_embed
#from .criterion import build_criterion

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, args, device):
        super(VisionTransformer, self).__init__(
            num_classes=args.classes_num,
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=args.in_chans,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            final_norm=True)

        self.args = args
        self.device = device
        self.model_name = 'ViT'
        self.img_size = args.img_size
        self.grid_size = args.img_size // args.patch_size

        #self.pos_embed.requires_grad = True

        self.freeze_backbone = args.freeze_backbone
        if self.freeze_backbone:
            self.head = torch.nn.Sequential(torch.nn.BatchNorm1d(self.head.in_features, affine=False, eps=1e-6), self.head)
            for name, param in self.named_parameters():
                if (not name.startswith('head')) and (not name.startswith('fc_norm')):
                    param.requires_grad = False

    def forward_features(self, img):
        x = self.patch_embed(img)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, img):
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.forward_features(img)
        else:
            x = self.forward_features(img)

        x = self.forward_head(x)

        return x


