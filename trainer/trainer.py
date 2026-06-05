import torch
# from torch.utils.data import DataLoader, sampler
# from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T

import time
import math
import os
import logging
import wandb
import matplotlib.pyplot as plt

from util.logger import initialize_exp, AverageMeter, RunningAverageMeter, sec_to_hm

import warnings

from models import NeighborMAE, MAE, SatMAEpp

import numpy as np
from PIL import Image

class Trainer():
    def __init__(self, args):
        self.args = args
        self.set_torch()
        self.build_logger()
        self.build_train_dataset()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()

        self.cur_iter = self.start_iter = 0
        self.cur_epoch = self.start_epoch = -1

        if args.resume_path is not None:
            self.resume_model()

    def set_torch(self):
        torch.set_num_threads(1)
        torch.manual_seed(self.args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(precision=10)

        self.enable_mixed_precision = True
        if self.args.bf16:
            self.precision = torch.bfloat16
        elif self.args.fp16:
            self.precision = torch.float16
        else:
            self.precision = torch.float32
            self.enable_mixed_precision = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_mixed_precision)

        self.rank = self.args.rank = int(os.environ["RANK"])
        self.local_rank = self.args.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = self.args.world_size = int(os.environ["WORLD_SIZE"])
        self.local_world_size = self.args.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        self.device = torch.device('cuda', self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend='nccl')

    def build_logger(self):
        self.logger = initialize_exp(self.args, self.rank)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        self.meter_name = ['data_time', 'batch_time', 'loss', 'aux_loss', 'weighted_loss', 'weighted_aux_loss', 'total_loss']
        self.meter = {name: RunningAverageMeter() for name in self.meter_name}

    def get_param_group(self):
        if self.args.filter_norm_bias:
            weight = []
            norm_and_bias = []
            for name, param in self.model.named_parameters():
                if 'norm' in name or 'bias' in name:
                    norm_and_bias.append(param)
                else:
                    weight.append(param)
            parameters = [{"params": weight, "weight_decay": self.args.wd},
                          {"params": norm_and_bias, "weight_decay": 0.0}]
        else:
            parameters = self.model.parameters()

        return parameters

    def build_optimizer(self):
        self.lr = self.args.base_lr * self.total_batch_size / 256
        parameters = self.get_param_group()
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            betas=self.args.betas)

        self.logger.info("Building optimizer done.")

    def build_scheduler(self):
        warmup_iter = self.total_iters * self.args.warmup_ratio
        cos_iter = max(1, self.total_iters - warmup_iter)
        lr_lambda = lambda cur_iter: (cur_iter + 1) / warmup_iter if cur_iter < warmup_iter else \
            (self.args.min_lr / self.lr + 0.5 * (1 - self.args.min_lr / self.lr) * \
            (1.0 + math.cos(math.pi * (cur_iter + 1 - warmup_iter) / cos_iter)))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.logger.info("Building scheduler done.")

    def build_model(self):
        if self.args.use_mae:
            model = MAE(self.args, self.device, self.total_iters)
        elif self.args.use_satmaepp:
            model = SatMAEpp(self.args, self.device, self.total_iters)
        else:
            model = NeighborMAE(self.args, self.device, self.total_iters)
        self.model_name = model.model_name

        model = model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.logger.info("Building {} model done.".format(self.model_name))

    def build_train_dataset(self):

        self.batch_size = self.args.batch_size
        self.loader_batch_size = int(round(self.batch_size / self.args.sample_num))
        self.total_batch_size = self.batch_size * self.args.world_size


        if self.args.dali or self.args.dali_cpu:
            os.environ["num_threads"] = str(self.args.workers)
            from data_loader.dali_loader_sim import build_dali_loader
            self.train_loader, self.dataset_size, self.actual_train_loader_len = build_dali_loader(self.args, self.loader_batch_size)
        else:
            def worker_init_fn(worker_id):
                if self.cpu_id is not None:
                    os.sched_setaffinity(0, [int(self.cpu_id[worker_id % len(self.cpu_id)])])

            from data_loader.torch_loader_sim import torchDataset
            self.train_dataset = torchDataset(self.args)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, sampler=self.train_sampler, batch_size=self.loader_batch_size, shuffle=False,
                worker_init_fn=worker_init_fn,
                num_workers=self.args.workers, pin_memory=True, drop_last=True,
                persistent_workers=(self.args.workers != 0))
            self.dataset_size = len(self.train_dataset)
            self.train_loader_len = len(self.train_loader)
            self.actual_train_loader_len = len(self.train_loader)


        self.train_loader_iterator = iter(self.train_loader)
        self.iters_per_epoch = self.dataset_size // self.total_batch_size
        self.epochs = self.args.epochs
        self.total_iters = self.iters_per_epoch * self.epochs
        self.save_interval = self.iters_per_epoch * self.args.save_interval
        #self.eval_interval = self.iters_per_epoch * self.args.eval_interval
        self.logger.info("Building {} data loader done with {} images loaded.".format("DALI"  if self.args.dali or self.args.dali_cpu else "PyTorch", self.dataset_size))


    def train(self):
        self.model.train()
        end_time = time.time()
        for it in range(0, self.total_iters - self.start_iter):
            if it % self.actual_train_loader_len == 0:
                self.train_loader_iterator = iter(self.train_loader)

            if self.cur_iter % self.iters_per_epoch == 0:
                self.cur_epoch = self.cur_epoch + 1
                self.logger.info("============ Starting epoch %i ... ============" % self.cur_epoch)

            data = next(self.train_loader_iterator)
            img, meta = self.preprocess_data(data)

            self.meter['data_time'].update(time.time() - end_time)

            with torch.amp.autocast('cuda', enabled=self.enable_mixed_precision, dtype=self.precision):
                loss, weighted_loss = self.model(img, meta)
                total_loss = weighted_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            self.meter['loss'].update(loss.item())
            self.meter['weighted_loss'].update(weighted_loss.item())

            if self.args.wandb_log and self.rank == 0:
                self.upload_wandb()

            self.cur_iter = self.cur_iter + 1
            if self.cur_iter % self.args.log_interval == 0:
                self.print_stats()
            if self.local_rank == 0 and self.cur_iter % self.save_interval == 0:
                self.save_model()

            self.meter['batch_time'].update(time.time() - end_time)
            end_time = time.time()

        self.save_model(suffix='final')

    def save_model(self, suffix=None):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.cur_epoch,
            "cur_iter": self.cur_iter,
            "args": self.args,
        }
        if suffix is None:
            suffix = f"epoch{self.cur_epoch}"
        torch.save(checkpoint, os.path.join(self.args.output_path, f"checkpoint_{suffix}.pth"))
        self.logger.info(f"Epoch {self.cur_epoch} | Training checkpoint <{suffix}> saved at {self.args.output_path}")

    def load_model(self):
        model_dict = torch.load(self.args.pretrained_path, map_location=self.device)
        if "model" in model_dict.keys():
            model_dict = model_dict["model"]
        self.model.module.load_state_dict(model_dict, strict=False)

    def resume_model(self):
        model_dict = torch.load(self.args.resume_path, map_location=self.device)
        self.model.module.load_state_dict(model_dict["model"])
        if not self.args.load_model_only:
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
            self.scaler.load_state_dict(model_dict["scaler"])
            self.cur_epoch = self.start_epoch = model_dict["epoch"]
            self.cur_iter = self.start_iter = model_dict["cur_iter"]
        self.logger.info(f"Loaded model and training status from {self.args.resume_path}. Start training from epoch {self.start_epoch}")

    def upload_wandb(self):
        wandb_log_dict = {'lr': self.optimizer.param_groups[0]['lr']}  # | log_info
        for k, v in self.meter.items():
            wandb_log_dict[k + '.val'] = v.val
            wandb_log_dict[k + '.avg'] = v.avg
        wandb.log(wandb_log_dict)

    def print_stats(self):
        batch_idx = self.cur_iter % self.iters_per_epoch
        left_time_this_epoch = sec_to_hm((self.iters_per_epoch - batch_idx) * self.meter['batch_time'].avg)
        left_time_all = sec_to_hm((self.iters_per_epoch * (self.epochs - self.cur_epoch - 1) + (self.iters_per_epoch - batch_idx)) * self.meter['batch_time'].avg)
        basic_info = (
            "Epoch [{epoch}-{batch_idx}/{iters_per_epoch}]\t"
            "Time [{left_time_all}|{left_time_this_epoch}]\t"
            "{batch_time.val:.3f}|{data_time.val:.3f} ({batch_time.avg:.3f}|{data_time.avg:.3f})\n"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "w_Loss {weighted_loss.val:.4f} ({weighted_loss.avg:.4f})\t"
            "lr {lr:.5f}"
            .format(
                epoch=self.cur_epoch,
                batch_idx=batch_idx,
                iters_per_epoch=self.iters_per_epoch,
                left_time_this_epoch=left_time_this_epoch,
                left_time_all=left_time_all,
                batch_time=self.meter['batch_time'],
                data_time=self.meter['data_time'],
                loss=self.meter['loss'],
                weighted_loss=self.meter['weighted_loss'],
                lr=self.optimizer.param_groups[-1]['lr']
            ))

        self.logger.info(basic_info)

    def preprocess_data(self, data):
        #B, K, C, H, W = img.shape
        img = [data[0]['img_{}'.format(i)] for i in range(self.args.sample_num)]
        meta = data[0]['meta']

        #idx = meta[:, :, 0].view(dtype=torch.long).to(self.device)
        left, top, right, bottom = meta[:, :, 0], meta[:, :, 1], meta[:, :, 2], meta[:, :, 3]

        x_min = left.min(dim=1, keepdim=True)[0]
        y_min = bottom.min(dim=1, keepdim=True)[0]

        left, right = left - x_min, right - x_min
        top, bottom = top - y_min, bottom - y_min

        x_max = right.max(dim=1, keepdim=True)[0]
        y_max = top.max(dim=1, keepdim=True)[0]

        left, right = left / x_max * 32, right / x_max * 32
        top, bottom = top / y_max * 32, bottom / y_max * 32

        pos = torch.stack([left, top, right, bottom], dim=-1)
        new_meta = {'pos': pos}

        return img, new_meta
