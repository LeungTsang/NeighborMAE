from trainer import Trainer
import argparse
import wandb
import os


parser = argparse.ArgumentParser(description="Implementation of NeighborMAE")

# experiment
parser.add_argument("--exp_name", type=str, default=None,
                    help="")
parser.add_argument("--output_path", type=str, required=True,
                    help="path to save model/logs")
parser.add_argument("--resume_path", type=str,
                    help="path to a model to resume training")
parser.add_argument("--load_model_only", action="store_true",
                    help="only load model weight when resuming")
parser.add_argument("--seed", default=3, type=int,
                    help="random seed")


# data
parser.add_argument("--data_path", type=str, required=True,
                    help="path to image")
parser.add_argument("--img_ext", type=str, default='jpg',
                    help="image extension")
parser.add_argument('--mean', default=[0.4203977400, 0.4231229357, 0.3936558223], type=float, nargs="+",
                    help='data mean')
parser.add_argument('--std', default=[0.1936633298, 0.1869270853, 0.1883962587], type=float, nargs="+",
                    help='data std')
parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers")

# training
parser.add_argument("--epochs", default=10, type=int,
                    help="number of total epochs to run")
parser.add_argument("--total_iters", default=-1, type=int,
                    help="")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--warmup_ratio", default=0.05, type=float,
                    help="")
parser.add_argument("--start_warmup_lr", default=0, type=float,
                    help="initial warmup learning rate")
parser.add_argument("--base_lr", default=1e-4, type=float,
                    help="base learning rate")
parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="final learning rate in cosine lr scheduler")
parser.add_argument('--betas', default=(0.9, 0.95), type=tuple, nargs="+",
                    help='adamw betas')
parser.add_argument("--wd", default=0.05, type=float,
                    help="weight decay")
parser.add_argument("--filter_norm_bias", action='store_true',
                    help="use f16 mixed precision")
parser.add_argument("--fp16", action='store_true',
                    help="use f16 mixed precision")
parser.add_argument("--bf16", action='store_true',
                    help="use bf16 mixed precision")
parser.add_argument("--compile", action='store_true',
                    help="use torch compile")

# log
parser.add_argument("--save_interval", default=5, type=int,
                    help="checkpoint save frequency")
parser.add_argument("--log_interval", default=10, type=int,
                    help="print training info frequency")
parser.add_argument("--eval_interval", default=10, type=int,
                    help="print training info frequency")
parser.add_argument("--wandb_log", action="store_true",
                    help="log to wandb")

# MAE specific:
parser.add_argument("--sample_num", default=2, type=int,
                    help="")
parser.add_argument("--crop_scale", default=[0.2, 1.0], type=float, nargs="+",
                    help="")
parser.add_argument("--aspect_ratio", default=0.75, type=float,
                    help="")
parser.add_argument('--img_size', default=224, type=int,
                    help='')
parser.add_argument('--patch_size', default=16, type=int,
                    help='')
parser.add_argument('--in_chans', default=3, type=int,
                    help='')
parser.add_argument('--mask_ratio_max', default=0.75, type=float,
                    help='')
parser.add_argument('--mask_ratio_min', default=0.75, type=float,
                    help='')
parser.add_argument("--embed_dim", default=768, type=int,
                    help="")
parser.add_argument("--depth", default=12, type=int,
                    help="")
parser.add_argument("--num_heads", default=12, type=int,
                    help="")
parser.add_argument("--decoder_embed_dim", default=512, type=int,
                    help="")
parser.add_argument("--decoder_depth", default=8, type=int,
                    help="")
parser.add_argument("--decoder_num_heads", default=16, type=int,
                    help="")
parser.add_argument("--mlp_ratio", default=4.0, type=float,
                    help="")
parser.add_argument("--norm_pix_loss", action="store_true",
                    help="")
parser.add_argument("--eval_only", action="store_true",
                    help="")
parser.add_argument("--use_mae", action="store_true",
                    help="")
parser.add_argument("--use_satmaepp", action="store_true",
                    help="")

# dali
parser.add_argument("--dali", action="store_true",
                    help="use dali data loader")
parser.add_argument("--dali_cpu", action="store_true",
                    help="use dali data loader of cpu version")

# distributed training
parser.add_argument('--rank', default=-1,
                    help='rank of current process')
parser.add_argument('--local_rank', default=-1,
                    help='local rank of current process')
parser.add_argument('--world_size', default=1,
                    help="world size")
parser.add_argument('--local_world_size', default=1,
                    help="local world size")
parser.add_argument('--init_method', default='tcp://localhost:10111',
                    help="url for distributed training")

args = parser.parse_args()
args.sample_num = 1 if args.use_mae or args.use_satmaepp else 2


if __name__ == "__main__":
    print("start")
    os.makedirs(args.output_path, exist_ok=True)
    if int(os.environ["RANK"]) == 0:
        if args.wandb_log:
            os.environ["WANDB__SERVICE_WAIT"] = "300"
            wandb.init(project="NeighborMAE", config=args, name=args.exp_name)

    trainer = Trainer(args)
    trainer.train()
    if args.wandb_log:
        wandb.finish()