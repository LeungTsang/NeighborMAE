import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
import os
import math

Image.MAX_IMAGE_PIXELS = None


class RandomResizedCrop(T.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = TF.get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w

class EvalDataset(data.Dataset):
    def __init__(self, args, is_train=False):
        super(EvalDataset, self).__init__()

        self.args = args
        self.file_root = args.data_path

        self.img_size = args.img_size
        self.mean = torch.tensor(args.mean, dtype=torch.float)
        self.std = torch.tensor(args.std, dtype=torch.float)
        self.aspect_ratio = (args.aspect_ratio, 1/args.aspect_ratio)

        if is_train:
            with open(os.path.join(self.args.data_path, 'split', self.args.split), 'r') as f:
                self.file_list = [line.rstrip() for line in f]

            self.transform = T.Compose([
                RandomResizedCrop(size=args.img_size, scale=args.crop_scale, ratio=self.aspect_ratio, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])

        else:
            with open(os.path.join(self.args.data_path, 'split', self.args.val_split), 'r') as f:
                self.file_list = [line.rstrip() for line in f]

            self.transform = T.Compose([
                T.Resize(size=int(args.img_size * 8 / 7), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=args.img_size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img_path, target = self.file_list[index].split(' ')
        img = Image.open(os.path.join(self.file_root, img_path)).convert('RGB')
        target = torch.tensor(int(target), dtype=torch.long)

        aug_img = self.transform(img)

        output = [{'img': aug_img, 'target': target}]

        return output


