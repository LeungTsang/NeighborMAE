import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
import os

import glob
import random
import math

Image.MAX_IMAGE_PIXELS = None

class torchDataset(data.Dataset):
    def __init__(self, args):
        super(torchDataset, self).__init__()

        self.args = args
        self.sample_num = args.sample_num
        #self.actual_batch_size = actual_batch_size

        self.files = sorted(glob.glob(os.path.join(args.data_path, '**', '*.' + args.img_ext), recursive=True))

        self.image_id_in_folder = []
        self.image_id_to_folder = []
        cur_folder = None
        for i, p in enumerate(self.files):
            folder = os.path.dirname(p)
            if folder != cur_folder:
                cur_folder = folder
                self.image_id_in_folder.append([i, ])
            else:
                self.image_id_in_folder[-1].append(i)
            self.image_id_to_folder.append(len(self.image_id_in_folder) - 1)

        self.img_size = args.img_size * 2 if args.use_satmaepp else args.img_size

        self.ratio = (math.log(self.args.aspect_ratio), math.log(1 / self.args.aspect_ratio))



    def __len__(self):
        return len(self.files)

    def __getitem__(self, sample_idx):

        if self.sample_num == 1:
            img_1 = Image.open(os.path.join(self.files[sample_idx])).convert('RGB')
            img = [TF.pil_to_tensor(img_1),]
        else:
            sample_neighbor_idx = random.sample(self.image_id_in_folder[self.image_id_to_folder[sample_idx]], 1)[0]
            img_1 = Image.open(os.path.join(self.files[sample_idx])).convert('RGB')
            img_2 = Image.open(os.path.join(self.files[sample_neighbor_idx])).convert('RGB')
            img = [TF.pil_to_tensor(img_1), TF.pil_to_tensor(img_2)]

        h_original = torch.tensor([img_i.shape[1] for img_i in img])
        w_original = torch.tensor([img_i.shape[2] for img_i in img])

        aspect_ratio = torch.exp(torch.empty(size=(self.sample_num,)).uniform_(*self.ratio))
        max_h = w_original / aspect_ratio

        target_area = torch.empty(size=(self.sample_num,)).uniform_(*self.args.crop_scale)
        scale_ratio = torch.sqrt(target_area)
        h_target_size = scale_ratio * h_original
        h_target_size, _ = torch.min(torch.stack((h_target_size, max_h, h_original), dim=1), dim=1)
        w_target_size = h_target_size * aspect_ratio

        p = torch.empty(size=(self.sample_num, 2)).uniform_(0, 1)
        img = [TF.crop(img[i],
                       top=(p[i, 0]*(h_original[i]-h_target_size[i])).round_().long().item(),
                       left=(p[i, 1]*(w_original[i]-w_target_size[i])).round_().long().item(),
                       height=h_target_size[i].round_().long().item(),
                       width=w_target_size[i].round_().long().item()) for i in range(self.sample_num)]

        img = [TF.resize(img[i], size=[self.img_size, self.img_size]) for i in range(self.sample_num)]
        img = [TF.normalize(img[i] / 255., mean=self.args.mean, std=self.args.std) for i in range(self.sample_num)]

        left = (w_original - w_target_size) * p[:, 1]
        top = h_original - (h_original - h_target_size) * p[:, 0]
        right = left + w_target_size
        bottom = top - h_target_size

        meta = torch.stack((left / h_original, top / h_original, right / h_original, bottom / h_original), dim=1)

        output = {'img_{}'.format(i): img[i] for i in range(self.args.sample_num)}
        output['meta'] = meta
        return [output]


#def build_torch_loader(args, is_train=False):
#    return Dataset(args)

