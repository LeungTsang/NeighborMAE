import sys

import torch
import torch.nn.functional as F


if __name__ == "__main__":

    ckpt = torch.load(sys.argv[1], map_location='cpu')
    model = ckpt['model']
    args = ckpt['args']

    new_model = {}
    #print(obj.keys())
    for k in list(model.keys()):
        old_k = k
        if 'decoder' in k or 'meta' in k or 'mask_token' in k or 'patch_embed' in k or 'cur_iter' in k or 'pad_token' in k:
            continue

        print(old_k, "->", k)
        new_model[k] = model[old_k]

    new_model['patch_embed.proj.weight'] = model['patch_embed.weight'].reshape(-1, 16, 16, 3).permute(0, 3, 1, 2)
    new_model['patch_embed.proj.bias'] = model['patch_embed.bias']
    print('patch_embed.weight', '->', 'patch_embed.proj.weight')
    print('patch_embed.bias', '->', 'patch_embed.proj.bias')

    pos_embed = model['pos_embed']
    cls_pos_embed = pos_embed[:, :1]
    pos_embed = pos_embed[:, 1:].unflatten(dim=1, sizes=(20, 20))[:, :14, :14].flatten(1, 2)
    pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
    new_model['pos_embed'] = pos_embed


    #print(pos_embed.shape)



    torch.save(new_model, sys.argv[2])

