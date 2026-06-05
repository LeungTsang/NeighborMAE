import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torchvision.transforms as T

import time
import math
import os
import logging
import wandb
import matplotlib.pyplot as plt

from util.logger import initialize_exp, AverageMeter, RunningAverageMeter, sec_to_hm
from timm.data.mixup import Mixup

from models import VisionTransformer
from data_loader_torch import EvalDataset

import warnings


warnings.filterwarnings('ignore')


class Trainer():
    def __init__(self, args):
        self.args = args
        self.set_torch()
        self.build_logger()
        self.build_train_dataset()
        self.build_val_dataset()
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        #
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
        self.scaler = GradScaler(enabled=self.enable_mixed_precision)

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
        self.meter_name = ['data_time', 'batch_time', 'loss', 'acc']
        self.meter = {name: RunningAverageMeter() for name in self.meter_name}
        self.val_meter_name = ['val_data_time', 'val_batch_time', 'val_loss', 'val_acc']
        self.val_meter = {name: RunningAverageMeter() for name in self.val_meter_name}

        self.max_val_acc = 0.0
        self.max_acc_epoch = 0

    def build_train_dataset(self):
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.total_batch_size = self.batch_size * self.args.world_size

        if self.args.dali or self.args.dali_cpu:
            os.environ["num_threads"] = str(self.args.workers)
            from data_loader_dali import build_dali_loader_cls
            self.train_loader, self.dataset_size, self.actual_train_loader_len = build_dali_loader_cls(self.args, self.batch_size)
        else:
            self.train_dataset = EvalDataset(self.args, is_train=True)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False,
                num_workers=self.args.workers, pin_memory=True, drop_last=True,
                persistent_workers=(self.args.workers != 0))
            self.dataset_size = len(self.train_dataset)
            self.actual_train_loader_len = len(self.train_loader)


        self.iters_per_epoch = self.dataset_size // self.total_batch_size
        self.total_iters = self.iters_per_epoch * self.epochs

        self.epochs = self.args.epochs
        self.total_iters = self.iters_per_epoch * self.epochs
        self.save_interval = self.iters_per_epoch * self.args.save_interval
        self.eval_interval = self.iters_per_epoch * self.args.eval_interval

        self.mixup_active = self.args.mixup > 0 or self.args.cutmix > 0. or self.args.cutmix_minmax is not None
        if self.mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.args.mixup, cutmix_alpha=self.args.cutmix, cutmix_minmax=self.args.cutmix_minmax,
                prob=self.args.mixup_prob, switch_prob=self.args.mixup_switch_prob, mode=self.args.mixup_mode,
                label_smoothing=self.args.smoothing, num_classes=self.args.classes_num)
        else:
            self.mixup_fn = None

        self.logger.info("Built {} train data loader done with {} images loaded.".format("DALI" if self.args.dali or self.args.dali_cpu else "PyTorch", self.dataset_size))

    def build_val_dataset(self):

        self.val_dataset = EvalDataset(self.args, is_train=False)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, sampler=self.val_sampler, batch_size=self.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True, drop_last=False,
            persistent_workers=(self.args.workers != 0))
        self.val_dataset_size = len(self.val_dataset)
        self.val_loader_len = len(self.val_loader)
        self.val_actual_train_loader_len = len(self.val_loader)

        self.logger.info("Building {} val data loader done with {} images loaded.".format("PyTorch", self.val_dataset_size))

    def build_model(self):
        model = VisionTransformer(self.args, self.device)
        self.model_name = model.model_name

        model = model.to(self.device)
        self.classes_num = self.args.classes_num
        self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.model_without_ddp = self.model.module
        self.logger.info("Building {} model done.".format(self.model_name))

    def build_optimizer(self):
        self.lr = self.args.base_lr * self.total_batch_size / 256
        if not self.args.freeze_backbone:
            parameters = self.get_param_group()
        else:
            parameters = self.model.parameters()

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.args.wd,
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

    def preprocess_data(self, data, is_train=True):
        data = data[0]
        img = data['img'].to(self.device)
        target = data['target'].long().to(self.device)

        if is_train and self.mixup_active and self.mixup_fn is not None:
            img, target = self.mixup_fn(img, target)

        return img, target

    def train(self):
        self.model.train()
        end_time = time.time()
        #self.evaluate()
        for it in range(self.start_iter, self.total_iters):
            if self.cur_iter % self.actual_train_loader_len == 0:
                self.train_loader_iterator = iter(self.train_loader)

            if self.cur_iter % self.iters_per_epoch == 0:
                self.cur_epoch = self.cur_epoch + 1
                self.logger.info("============ Starting epoch %i ... ============" % self.cur_epoch)

            data = next(self.train_loader_iterator)
            img, target = self.preprocess_data(data)

            self.meter['data_time'].update(time.time() - end_time)

            with autocast(enabled=self.enable_mixed_precision, dtype=self.precision):
                logit = self.model(img)
                loss = F.cross_entropy(logit, target)
                pred = torch.argmax(logit, dim=1)
                if target.ndim > 1:
                    target = target.argmax(dim=1)
                acc = pred.eq(target).float().mean() * 100

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            self.meter['loss'].update(loss.item())
            self.meter['acc'].update(acc.item())

            if self.args.wandb_log and self.rank == 0:
                self.upload_wandb()

            self.cur_iter = self.cur_iter + 1
            if self.cur_iter % self.args.log_interval == 0:
                self.print_stats()
            if self.cur_iter % self.eval_interval == 0:
                self.evaluate()
                self.model.train()
            if self.local_rank == 0 and self.cur_iter % self.save_interval == 0:
                self.save_model()

            self.meter['batch_time'].update(time.time() - end_time)
            end_time = time.time()

        self.save_model(suffix='final')
        self.evaluate()

    def save_model(self, suffix=None):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.cur_epoch,
            "iter": self.cur_iter,
            "args": self.args,
        }
        if suffix is None:
            suffix = f"epoch{self.cur_epoch}_iter{self.cur_iter}"
        torch.save(checkpoint, os.path.join(self.args.output_path, f"checkpoint_{suffix}.pth"))
        self.logger.info(f"Epoch {self.cur_epoch} | Training checkpoint <{suffix}> saved at {self.args.output_path}")

    def load_model(self):
        model_dict = torch.load(self.args.pretrained_path, map_location=self.device)
        if "model" in model_dict.keys():
            model_dict = model_dict["model"]
            msg = self.model.module.load_state_dict(model_dict, strict=False)
        else:
            msg = self.model.module.load_state_dict(model_dict, strict=False)
        print(msg)
    def resume_model(self):
        model_dict = torch.load(self.args.resume_path, map_location=self.device)
        if "model" in model_dict.keys():
            msg = self.model.module.load_state_dict(model_dict["model"], strict=False)
            if not self.args.eval_only or not self.args.load_model_only:
                self.optimizer.load_state_dict(model_dict["optimizer"])
                self.lr_scheduler.load_state_dict(model_dict["lr_scheduler"])
                self.scaler.load_state_dict(model_dict["scaler"])
                self.start_epoch = model_dict["epoch"]
                self.cur_iter = model_dict["iter"]
        else:
            msg = self.model.module.load_state_dict(model_dict, strict=False)

        self.logger.info(f"Loaded model and training status from {self.args.resume_path}. Start training from epoch {self.start_epoch}")

    def upload_wandb(self):
        wandb_log_dict = {'lr': self.optimizer.param_groups[-1]['lr']}  # | log_info
        for k, v in self.meter.items():
            wandb_log_dict[k + '.val'] = v.val
            wandb_log_dict[k + '.avg'] = v.avg
        wandb.log(wandb_log_dict)

    def print_stats(self):
        batch_idx = self.cur_iter % self.iters_per_epoch
        left_time_this_epoch = sec_to_hm((self.iters_per_epoch - batch_idx) * self.meter['batch_time'].avg)
        left_time_all = sec_to_hm((self.iters_per_epoch * (self.epochs - self.cur_epoch - 1) + (self.iters_per_epoch - batch_idx)) * self.meter['batch_time'].avg)
        basic_info = (
            "Epoch [{batch_idx}/{iters_per_epoch}-{epoch}/{max_epoch}-{cur_iter}/{max_iter}]\t"
            "Time [{left_time_all}|{left_time_this_epoch}]\t"
            "{batch_time.val:.3f}|{data_time.val:.3f} ({batch_time.avg:.3f}|{data_time.avg:.3f})\n"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            "Acc {acc.val:.4f} ({acc.avg:.4f})\t"
            "lr {lr:.6f}"
            .format(
                epoch=self.cur_epoch,
                max_epoch=self.epochs,
                batch_idx=batch_idx,
                iters_per_epoch=self.iters_per_epoch,
                cur_iter=self.cur_iter,
                max_iter=self.total_iters,
                left_time_this_epoch=left_time_this_epoch,
                left_time_all=left_time_all,
                batch_time=self.meter['batch_time'],
                data_time=self.meter['data_time'],
                loss=self.meter['loss'],
                acc=self.meter['acc'],
                lr=self.optimizer.param_groups[-1]['lr']
            ))

        self.logger.info(basic_info)

    def evaluate(self):
        self.model.eval()

        for k in self.val_meter.keys():
            self.val_meter[k].reset()
        correct_k = torch.zeros(2, device=self.device, dtype=torch.long)
        val_loss = torch.zeros(1, device=self.device, dtype=torch.float64)

        end_time = time.time()
        for batch_idx, data in enumerate(self.val_loader):
            img, target = self.preprocess_data(data, is_train=False)

            self.val_meter['val_data_time'].update(time.time() - end_time)

            with torch.no_grad():
                bs = target.shape[0]
                logit = self.model(img)

                loss = F.cross_entropy(logit, target)
                pred = torch.argmax(logit, dim=1)
                if target.ndim > 1:
                    target = target.argmax(dim=1)
                acc = pred.eq(target).float().mean() * 100
                val_loss = val_loss + loss * bs

                topk = (1, min(5, self.classes_num))
                maxk = max(topk)

                _, pred = logit.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))

                for i, k in enumerate(topk):
                    correct_k[i] = correct_k[i] + correct[:k].reshape(-1).float().sum(0, keepdim=True)

                self.val_meter['val_loss'].update(loss.item(), bs)
                self.val_meter['val_acc'].update(acc.item(), bs)

                if (batch_idx + 1) % self.args.log_interval == 0:
                    left_time = sec_to_hm((self.val_actual_train_loader_len - batch_idx) * self.val_meter['val_batch_time'].avg)
                    basic_info = (
                        "Evaluating...\t"
                        "Batch [{batch_idx}/{batch_num}]\t"
                        "Left Time [{left_time}] ({batch_time.avg:.3f}|{data_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Acc {acc.val:.4f} ({acc.avg:.4f})\t"
                        .format(
                            batch_idx=batch_idx+1,
                            batch_num=self.val_actual_train_loader_len,
                            left_time=left_time,
                            batch_time=self.val_meter['val_batch_time'],
                            data_time=self.val_meter['val_data_time'],
                            loss=self.val_meter['val_loss'],
                            acc=self.val_meter['val_acc'],
                        ))

                    self.logger.info(basic_info)

                self.val_meter['val_batch_time'].update(time.time() - end_time)
                end_time = time.time()

        torch.distributed.all_reduce(correct_k, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.SUM)

        val_acc1 = correct_k[0].item() / self.val_dataset_size * 100
        val_acc5 = correct_k[1].item() / self.val_dataset_size * 100
        val_loss = val_loss.item() / self.val_dataset_size

        if val_acc1 > self.max_val_acc:
            self.max_val_acc = val_acc1
            self.max_acc_epoch = self.cur_epoch
            if (not self.args.eval_only) and self.rank == 0:
                self.save_model(suffix='best')

        if self.args.wandb_log and self.rank == 0:
            wandb_log_dict = {"val_Acc1": val_acc1, "val_Acc5": val_acc5, "val_loss": val_loss, "max_val_Acc": self.max_val_acc}
            wandb.log(wandb_log_dict)

        torch.distributed.barrier()

        result = (
            "\n====================================\n"
            "Eval Result at Epoch {epoch}: \n"
            "val_Loss: [{loss:.4f}]\t"
            "val_Acc1: [{acc1:.4f}]\t"
            "val_Acc5: [{acc5:.4f}]"
            "\n------------------------------------\n"
            "Max val_Acc1: [{max_acc1:.4f} (Epoch {max_epoch})]  \n"
            "\n====================================\n"
            .format(
                epoch=self.cur_epoch,
                loss=val_loss,
                acc1=val_acc1,
                acc5=val_acc5,
                max_acc1=self.max_val_acc,
                max_epoch=self.max_acc_epoch
            ))
        self.logger.info(result)
        self.model.train()

    def get_param_group(self):
        if self.args.lrd and not self.args.freeze_backbone:
            def get_layer_id_for_vit(name, num_layers):
                if name in ['cls_token', 'pos_embed', 'img_embed', 'backbone.cls_token', 'backbone.pos_embed', 'backbone.img_embed']:
                    return 0
                elif name.startswith('patch_embed') or name.startswith('backbone.patch_embed'):
                    return 0
                elif name.startswith('blocks') or name.startswith('backbone.blocks'):
                    name = name.replace('backbone.', '')
                    return int(name.split('.')[1]) + 1
                else:
                    return num_layers

            no_weight_decay_list = ['pos_embed', 'cls_token', 'dist_token', 'img_embed', 'backbone.pos_embed', 'backbone.cls_token', 'backbone.dist_token', 'backbone.img_embed']
            param_groups = {}
            param_group_names = {}

            if hasattr(self.model_without_ddp, 'backbone'):
                num_layers = len(self.model_without_ddp.backbone.blocks) + 1
            else:
                num_layers = len(self.model_without_ddp.blocks) + 1
            layer_scales = list(self.args.layer_decay ** (num_layers - i) for i in range(num_layers + 1))

            for name, param in self.model_without_ddp.named_parameters():
                if not param.requires_grad:
                    continue

                if param.ndim == 1 or name in no_weight_decay_list:
                    g_decay = "no_decay"
                    this_decay = 0.
                else:
                    g_decay = "decay"
                    this_decay = self.args.wd

                layer_id = get_layer_id_for_vit(name, num_layers)
                group_name = "layer_%d_%s" % (layer_id, g_decay)

                if group_name not in param_groups:
                    this_scale = layer_scales[layer_id]
                    param_groups[group_name] = {
                        "lr": self.lr * this_scale,
                        "weight_decay": this_decay,
                        "params": [],
                    }
                    param_group_names[group_name] = {
                        "lr": self.lr * this_scale,
                        "weight_decay": this_decay,
                        "params": [],
                    }

                param_groups[group_name]["params"].append(param)
                param_group_names[group_name]["params"].append(name)
            return list(param_groups.values())

        else:
            return self.model.parameters()


    @staticmethod
    def show_img(img):
        # if len(img.shape) == 5:
        #B, K, C, H, W = img.shape
        inv_normalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
        img = [inv_normalize(img_.cpu()).permute(0, 2, 3, 1).numpy() for img_ in img]
        B = img[0].shape[0]
        K = len(img)

        for i in range(B):
            for j in range(K):
                plt.subplot(1, K, j + 1)
                plt.imshow(img[j][i])
                plt.axis('off')
                plt.tight_layout()
            plt.show()

