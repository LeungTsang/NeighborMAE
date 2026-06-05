# --------------------------------------------------------
# References:
# MAE： https://github.com/facebookresearch/mae
# --------------------------------------------------------


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from functools import partial

from .block import Block


class NeighborMAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, args, device, total_iters):
        super().__init__()

        self.model_name = 'NeighborMAE'

        self.args = args
        self.device = device
        self.total_iters = total_iters

        self.batch_size = args.batch_size
        self.crop_num = 2

        # input
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.grid_size = args.img_size // args.patch_size
        self.mask_ratio_max = args.mask_ratio_max
        self.mask_ratio_min = args.mask_ratio_min
        self.num_patches = (args.img_size // args.patch_size) ** 2
        self.total_num_patches = self.num_patches * self.crop_num #sum(self.num_patches)
        self.len_keep = int(round((1 - self.mask_ratio_min) * self.num_patches))
        self.input_dim = args.patch_size * args.patch_size * args.in_chans

        # model
        self.embed_dim = args.embed_dim
        self.decoder_embed_dim = args.decoder_embed_dim
        self.norm_pix_loss = args.norm_pix_loss
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = nn.Linear(self.input_dim, args.embed_dim, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            Block(args.embed_dim, args.num_heads, args.mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(args.depth)])
        self.norm = norm_layer(args.embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = args.decoder_embed_dim
        self.in_chans = args.in_chans

        self.decoder_embed = nn.Linear(args.embed_dim, args.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim), requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            Block(args.decoder_embed_dim, args.decoder_num_heads, args.mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(args.decoder_depth)])

        self.decoder_norm = norm_layer(args.decoder_embed_dim)
        self.decoder_pred = nn.Linear(args.decoder_embed_dim, self.input_dim, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # meta embedding specifics
        self.img_embed = nn.Parameter(torch.zeros(1, self.crop_num, args.embed_dim), requires_grad=True)
        self.decoder_img_embed = nn.Parameter(torch.zeros(1, self.crop_num, args.decoder_embed_dim), requires_grad=True)

        self.register_buffer("cur_iter", torch.zeros(1, dtype=torch.long))
        self.initialize_weights()


        step = torch.linspace(-1 + 1 / self.img_size, 1 - 1 / self.img_size, self.img_size, dtype=torch.float, device=self.device)
        self.grid = torch.stack(torch.meshgrid(step, step, indexing='xy'), dim=-1).view(1, 1, self.img_size, self.img_size, 2)

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_img_embed, std=.02)
        torch.nn.init.normal_(self.img_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, img, p, c=3):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        #p = self.patch_embed.patch_size[0]
        assert img.shape[-2] == img.shape[-1] and img.shape[-1] % p == 0

        h = w = img.shape[-1] // p
        x = img.view(-1, c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(*img.shape[:-3], h * w, p ** 2 * c))
        return x

    def unpatchify(self, x, p, c=3):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        h = w = int(x.shape[-2] ** .5)
        assert h * w == x.shape[-2]

        imgs = x.view(-1, h, w, p, p, c)
        imgs = torch.einsum('nhwpqc->nchpwq', imgs)
        imgs = imgs.reshape(*x.shape[:-2], c, h * p, w * p)
        return imgs


    def patchify_pos(self, pos):
        B, K, _ = pos.shape

        xy = pos.view(-1, 2, 2).transpose(1, 2)
        xy = F.interpolate(xy, size=self.grid_size + 1, mode='linear', align_corners=True)
        x = xy[:, 0].unsqueeze(1).expand(-1, self.grid_size + 1, -1)
        y = xy[:, 1].unsqueeze(2).expand(-1, -1, self.grid_size + 1)

        left_patch, right_patch = x[:, :-1, :-1], x[:, 1:, 1:]
        top_patch, bottom_patch = y[:, :-1, :-1], y[:, 1:, 1:]

        patch_pos = torch.stack([left_patch, top_patch, right_patch, bottom_patch], dim=-1)
        patch_pos = patch_pos.view(B, K, -1, 4)

        return patch_pos

    def pairwise_miou(self, pos):
        left, top, right, bottom = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2], pos[:, :, 3]  # B K

        left_i = torch.maximum(left.unsqueeze(2), left.unsqueeze(1))
        right_i = torch.minimum(right.unsqueeze(2), right.unsqueeze(1))
        top_i = torch.minimum(top.unsqueeze(2), top.unsqueeze(1))
        bottom_i = torch.maximum(bottom.unsqueeze(2), bottom.unsqueeze(1))

        intersection = (right_i - left_i).clamp(min=0) * (top_i - bottom_i).clamp(min=0)
        area = (right - left) * (top - bottom)
        union = area.unsqueeze(2) + area.unsqueeze(1) - intersection

        iou = intersection / union
        miou = (iou.sum(dim=2) - 1) / (self.crop_num - 1)

        return miou

    def make_pos_embed(self, pos):
        B, K, _ = pos.shape
        patch_pos = self.patchify_pos(pos)
        encoder_pos_embed = self.sincos_pos_embed(patch_pos, self.embed_dim // 4, w=10000).flatten(3, 4)
        encoder_pos_embed = encoder_pos_embed + self.img_embed.unsqueeze(2)

        decoder_pos_embed = self.sincos_pos_embed(patch_pos, self.decoder_embed_dim // 4, w=10000).flatten(3, 4)
        decoder_pos_embed = decoder_pos_embed + self.decoder_img_embed.unsqueeze(2)

        return encoder_pos_embed, decoder_pos_embed


    def proj(self, img, visible_mask, pos):

        pos = pos.unflatten(dim=-1, sizes=(2, 2))
        center = pos.mean(dim=-2)  # B K 2
        size = torch.abs(pos[:, :, 1] - pos[:, :, 0])  # B K 2

        translate = (center - center[:, [1, 0]]) * (2 / size[:, [1, 0]])  # B K 2
        translate[:, :, 1] = -translate[:, :, 1]

        scale = size / size[:, [1, 0]]  # B K 2
        grid = self.grid * scale.unsqueeze(2).unsqueeze(2)
        grid = grid + translate.unsqueeze(2).unsqueeze(2)

        img = img[:, [1, 0]].flatten(0, 1)
        img_proj = F.grid_sample(img, grid.flatten(0, 1), mode='bilinear', align_corners=False)
        img_proj = img_proj.unflatten(dim=0, sizes=(-1, self.crop_num))

        visible_mask = visible_mask[:, [1, 0]].float().flatten(0, 1)
        visible_mask_proj = F.grid_sample(visible_mask, grid.flatten(0, 1), mode='nearest', align_corners=False)
        visible_mask_proj = visible_mask_proj.unflatten(dim=0, sizes=(-1, self.crop_num)).bool()

        return img_proj, visible_mask_proj


    def forward(self, img, meta):
        self.update_training_status()


        # stack neighboring images
        img = torch.stack(img, dim=1)  # B K 3 H W
        patches = self.patchify(img, p=self.patch_size, c=self.in_chans)
        N, S, L, D = patches.shape

        # joint relative position embedding
        encoder_pos_embed, decoder_pos_embed = self.make_pos_embed(meta['pos'])
        encoder_pos_embed, decoder_pos_embed = encoder_pos_embed, decoder_pos_embed

        # dynamic mask ratio based on iou
        miou = self.pairwise_miou(meta['pos'])

        mask_ratio = self.mask_ratio_min + miou * (self.mask_ratio_max - self.mask_ratio_min)
        len_keep = ((1 - mask_ratio) * self.num_patches).round_().long()

        token_ids = torch.arange(0, self.num_patches, device=self.device)  # .expand(N, -1)
        binary_mask = token_ids.unsqueeze(0).unsqueeze(0) < len_keep.unsqueeze(2)

        noise = torch.rand(size=(N, self.crop_num, self.num_patches), device=self.device)
        ids_shuffle = torch.argsort(noise, dim=2)
        ids_restore = torch.argsort(ids_shuffle, dim=2)
        ids_keep = ids_shuffle[:, :, :self.len_keep]

        # prepare encoder input
        patches_keep = torch.gather(patches, dim=2, index=ids_keep.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        encoder_pos_embed = torch.gather(encoder_pos_embed, dim=2, index=ids_keep.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim))

        x = self.patch_embed(patches_keep)
        x = x + encoder_pos_embed
        x = x.flatten(1, 2)
        x = torch.cat((self.cls_token.expand(N, -1, -1), x), dim=1)

        # mask patches more than min mask ratio by attention mask
        attn_mask = binary_mask[:, :, :self.len_keep].reshape(N, 1, 1, -1)
        attn_mask = torch.cat((torch.ones(size=(N, 1, 1, 1), dtype=torch.bool, device=self.device), attn_mask), dim=3)

        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask)
        x = self.norm(x)
        x = self.decoder_embed(x)

        # prepare decoder input
        cls = x[:, :1]
        x = x[:, 1:]

        x = x.unflatten(dim=1, sizes=(self.crop_num, self.len_keep))
        x = torch.cat((self.mask_token.unsqueeze(2).expand(N, self.crop_num, -1, -1), x), dim=2)
        ids_mask = token_ids.repeat(N, S, 1) + 1
        ids_mask[~binary_mask] = 0

        x = torch.gather(x, dim=2, index=ids_mask.unsqueeze(-1).expand(-1, -1, -1, self.decoder_embed_dim))
        x = torch.cat((cls, x.flatten(1, 2)), dim=1)

        decoder_pos_embed = torch.gather(decoder_pos_embed, dim=2, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, -1, self.decoder_embed_dim))
        cls_embed = torch.zeros(size=(N, 1, self.decoder_embed_dim), device=self.device)
        decoder_pos_embed = torch.cat((cls_embed, decoder_pos_embed.flatten(1, 2)), dim=1)

        x = x + decoder_pos_embed

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)

        x = x[:, 1:]
        x = x.unflatten(dim=1, sizes=(self.crop_num, self.num_patches))
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)

        # resume image layout

        pred = torch.gather(pred, dim=2, index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, self.input_dim))
        pred = self.unpatchify(pred, p=self.patch_size, c=self.in_chans)

        visible_mask = torch.gather(binary_mask, dim=2, index=ids_restore)
        visible_mask = visible_mask.unsqueeze(-1).repeat(1, 1, 1, self.patch_size ** 2)
        visible_mask = self.unpatchify(visible_mask, p=self.patch_size, c=1)
        invisible_mask = ~visible_mask

        # cross project images
        img_proj, visible_mask_proj = self.proj(img, visible_mask, meta['pos'].float())

        if self.norm_pix_loss:
            img_stat = img.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
            mean = img_stat.mean(dim=(2, 5, 6)).repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3).unsqueeze(2)
            var = img_stat.var(dim=(2, 5, 6)).repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3).unsqueeze(2)
            target = (img - mean) / (var + 1.e-6) ** .5
            img_proj = (img_proj - mean) / (var + 1.e-6) ** .5
        else:
            target = img
            img_proj = img_proj

        # compute loss weight by cross visibility
        loss = F.mse_loss(pred, target, reduction='none').mean(dim=2, keepdim=True)
        img_proj_loss = F.mse_loss(img_proj, target, reduction='none').mean(dim=2, keepdim=True)

        loss_weight = (img_proj_loss / loss.detach()).clamp(0, 1.)
        loss_weight.masked_fill_(~visible_mask_proj, 1)
        loss_weight = F.avg_pool2d(loss_weight.flatten(0, 1), kernel_size=9, stride=1, padding=4, count_include_pad=False).unflatten(dim=0, sizes=(N, self.crop_num))
        loss_weight.masked_fill_(visible_mask, 0)

        weighted_loss = (loss * loss_weight).sum() / invisible_mask.sum()
        loss = (loss * invisible_mask).sum() / invisible_mask.sum()

        return loss, weighted_loss


    @torch.no_grad()
    def update_training_status(self):
        self.cur_iter += 1

    @staticmethod
    def sincos_pos_embed(pos, dim, w):

        sub_dim = dim // 2
        omega = torch.linspace(0, 1, sub_dim, device=pos.device, dtype=pos.dtype)
        # omega /= sub_dim
        omega = 1. / w ** omega

        out = torch.einsum('..., d->...d', pos, omega)  # pos.view(B, K, -1, 1) * omega.view(1, 1, 1, -1)

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (M, D)

        return emb.float()

