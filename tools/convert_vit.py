import sys

import torch
import torch.nn.functional as F


def patchify_pos(pos, grid_size):
    B, K, _ = pos.shape
    xy = pos.view(B * K, 2, 2).transpose(1, 2)  # lon lat lon lat
    xy = F.interpolate(xy, size=grid_size + 1, mode='linear', align_corners=True)
    xy = xy.view(B, K, 2, grid_size + 1)

    x = xy[:, :, 0].unsqueeze(2).expand(-1, -1, grid_size + 1, -1)
    y = xy[:, :, 1].unsqueeze(3).expand(-1, -1, -1, grid_size + 1)

    left_patch, right_patch = x[:, :, :-1, :-1], x[:, :, 1:, 1:]
    top_patch, bottom_patch = y[:, :, :-1, :-1], y[:, :, 1:, 1:]

    patch_pos = torch.stack([left_patch, top_patch, right_patch, bottom_patch], dim=-1)
    patch_pos = patch_pos.flatten(2, 3)

    return patch_pos


def make_default_pos_embed(embed_dim, img_embed):
    pos = torch.tensor([0., 32., 32., 0.], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    patch_pos = patchify_pos(pos)
    patch_pos_embed = sincos_pos_embed(patch_pos, embed_dim // 4, w=10000).flatten(3, 4)
    pos_embed = patch_pos_embed + img_embed[:, :1, :, :]

    return pos_embed


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


if __name__ == "__main__":

    ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    model = ckpt['model']

    new_model = {}
    for k in list(model.keys()):
        old_k = k
        if 'decoder' in k or 'mask_token' in k or 'patch_embed' in k or 'pos_embed' in k or 'cur_iter' in k or 'pad_token' in k or 'img_embed' in k:
            print('skip {}'.format(k))
            continue

        print(old_k, "->", k)
        new_model[k] = model[old_k]

    new_model['patch_embed.proj.weight'] = model['patch_embed.weight'].reshape(-1, 16, 16, 3).permute(0, 3, 1, 2)
    new_model['patch_embed.proj.bias'] = model['patch_embed.bias']
    print('patch_embed.weight', '->', 'patch_embed.proj.weight')
    print('patch_embed.bias', '->', 'patch_embed.proj.bias')

    grid_size = 224 // 16
    dim = new_model['patch_embed.proj.weight'].shape[0]

    pos = torch.tensor([0., 32., 32., 0.], dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    patch_pos = patchify_pos(pos, grid_size)
    pos_embed = sincos_pos_embed(patch_pos, dim // 4, w=10000).flatten(3, 4).squeeze(1).float()

    pos_embed = pos_embed + model['img_embed'][:, :1]
    pos_embed = torch.cat((torch.zeros(1, 1, dim), pos_embed), dim=1)

    new_model['pos_embed'] = pos_embed

    print("make default pos embed for single image input with grid size {}x{}".format(grid_size, grid_size))

    torch.save(new_model, sys.argv[2])
