# --------------------------------------------------------
# References:
# MAE： https://github.com/facebookresearch/mae
# --------------------------------------------------------



import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed


class MAE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, args, device, total_iters):
        super().__init__()

        self.model_name = 'MAE'

        self.args = args
        self.device = device
        self.total_iters = total_iters
        # self.is_train = is_train

        # batch
        self.batch_size = args.batch_size

        # input
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.grid_size = args.img_size // args.patch_size
        self.num_patches = (args.img_size // args.patch_size) ** 2
        self.len_keep = int(round(self.num_patches * (1 - args.mask_ratio_min)))
        self.len_remove = self.num_patches - self.len_keep
        self.input_dim = args.patch_size * args.patch_size * args.in_chans

        # model
        self.embed_dim = args.embed_dim
        self.decoder_embed_dim = args.decoder_embed_dim
        self.mask_ratio = args.mask_ratio_min
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
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, args.embed_dim), requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, args.decoder_embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(args.embed_dim, self.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(args.decoder_embed_dim, self.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.register_buffer("cur_iter", torch.zeros(1, dtype=torch.long))
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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
        # p = self.patch_embed.patch_size[0]
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


    def forward(self, img, meta=None):
        self.update_training_status()

        img = torch.stack(img, dim=1).flatten(0, 1)  # B K 3 H W
        patches = self.patchify(img, p=self.patch_size, c=self.in_chans)
        N, L, D = patches.shape

        encoder_pos_embed = self.pos_embed.expand(N, -1, -1)
        decoder_pos_embed = self.decoder_pos_embed.expand(N, -1, -1)

        noise = torch.rand(size=(N, self.num_patches), device=self.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :self.len_keep]

        patches_keep = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.input_dim))
        encoder_pos_embed = torch.gather(encoder_pos_embed, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        x = self.patch_embed(patches_keep)
        x = x + encoder_pos_embed
        x = torch.cat((self.cls_token.expand(N, -1, -1), x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        x = self.decoder_embed(x)

        decoder_pos_embed = torch.gather(decoder_pos_embed, dim=1, index=ids_shuffle.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim))
        cls_embed = torch.zeros(size=(N, 1, self.decoder_embed_dim), device=self.device)
        decoder_pos_embed = torch.cat((cls_embed, decoder_pos_embed), dim=1)
        x = torch.cat([x, self.mask_token.expand(N, self.len_remove, -1)], dim=1)
        x = x + decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        pred = pred[:, self.len_keep + 1:, :]

        target = torch.gather(patches, dim=1, index=ids_shuffle[:, self.len_keep:].unsqueeze(-1).expand(-1, -1, self.input_dim))

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = F.mse_loss(pred, target)
        weighted_loss = loss

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
