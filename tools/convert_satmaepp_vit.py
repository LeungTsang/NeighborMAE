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
        if 'decoder' in k or 'meta' in k or 'mask_token' in k or 'cur_iter' in k or 'pad_token' in k or 'up' in k:
            continue

        print(old_k, "->", k)
        new_model[k] = model[old_k]

    torch.save(new_model, sys.argv[2])

