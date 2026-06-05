from trainer import Trainer
import argparse
import wandb
import os


parser = argparse.ArgumentParser(description="Implementation of GMAE")

# init
parser.add_argument("--model_name", type=str,
                    help="model name")
parser.add_argument("--data_path", type=str,
                    help="path to image")
# parser.add_argument("--stat", type=str, default=None,
#                     help="norm stat alias")
parser.add_argument('--mean', default=[0.4203977400, 0.4231229357, 0.3936558223], type=float, nargs="+",
                    help='')
parser.add_argument('--std', default=[0.1936633298, 0.1869270853, 0.1883962587], type=float, nargs="+",
                    help='')
parser.add_argument("--split", type=str,
                    help="data split name")
parser.add_argument("--val_split", type=str,
                    help="data split name for validation")
parser.add_argument("--output_path", type=str,
                    help="path to save model/logs")

parser.add_argument("--resume_path", type=str,
                    help="path to model to resume training")
# parser.add_argument("--pretrained_path", type=str,
#                     help="path to load a pretrained model")
parser.add_argument("--load_model_only", action="store_true",
                    help="")
parser.add_argument("--seed", default=3, type=int,
                    help="randome seed")
parser.add_argument("--workers", default=8, type=int,
                    help="number of data loading workers")
parser.add_argument("--exp_name", type=str, default=None,
                    help="")


# training
parser.add_argument("--epochs", default=10, type=int,
                    help="number of total epochs to run")
parser.add_argument("--total_iters", default=-1, type=int,
                    help="number of total iters to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--val_batch_size", default=-1, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--warmup_ratio", default=0.05, type=float,
                    help="")
parser.add_argument("--start_warmup_lr", default=0, type=float,
                    help="initial warmup learning rate")
parser.add_argument("--base_lr", default=1e-4, type=float,
                    help="base learning rate")
parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="final learning rate in cosine lr scheduler")
parser.add_argument("--wd", default=0.05, type=float,
                    help="weight decay")
parser.add_argument("--filter_norm_bias", action='store_true',
                    help="use f16 mixed precision")
parser.add_argument("--fp16", action='store_true',
                    help="use f16 mixed precision")
parser.add_argument("--bf16", action='store_true',
                    help="use bf16 mixed precision")
parser.add_argument("--lrd", action="store_true",
                    help="use layer wise learning rate decay")
parser.add_argument("--layer_decay", default=0.65, type=float,
                    help="layer wise learning rate decay rate")
parser.add_argument('--betas', default=[0.9, 0.999], type=float, nargs="+",
                    help='adamw betas')

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
parser.add_argument("--classes_num", default=62, type=int,
                    help="")
parser.add_argument('--img_size', default=224, type=int,
                    help='')
parser.add_argument("--crop_scale", default=[0.2, 1.0], type=float, nargs="+",
                    help="")
parser.add_argument("--aspect_ratio", default=0.75, type=float,
                    help="")
parser.add_argument('--patch_size', default=16, type=int,
                    help='')
parser.add_argument('--in_chans', default=3, type=int,
                    help='')
parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='')
parser.add_argument("--embed_dim", default=768, type=int,
                    help="")
parser.add_argument("--depth", default=12, type=int,
                    help="")
parser.add_argument("--num_heads", default=12, type=int,
                    help="")
parser.add_argument("--mlp_ratio", default=4.0, type=float,
                    help="")
parser.add_argument("--mode", default=0, type=int,
                    help="")
parser.add_argument("--pos_mode", default=0, type=int,
                    help="")
parser.add_argument("--data_mode", default=0, type=int,
                    help="")

parser.add_argument("--global_pool", default='avg', type=str,
                    help="")
parser.add_argument("--drop_path", default=0.1, type=float,
                    help="")


parser.add_argument("--freeze_backbone", action="store_true",
                    help="")
parser.add_argument("--eval_only", action="store_true",
                    help="")

# * Mixup params
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

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

if __name__ == "__main__":
    # known_mean = {
    #     'fmow': [0.4203977400, 0.4231229357, 0.3936558223],
    #     'satellogic': [0.3958018576, 0.3558682431, 0.3113365802],
    # }
    # known_std = {
    #     'fmow': [0.1936633298, 0.1869270853, 0.1883962587],
    #     'satellogic': [0.1180706066, 0.1051147621, 0.0943769392],
    # }
    # if args.stat is not None:
    #     args.mean, args.std = known_mean[args.stat], known_std[args.stat]

    # print("start")
    os.makedirs(args.output_path, exist_ok=True)
    if int(os.environ["RANK"]) == 0:
        if args.wandb_log:
            os.environ["WANDB__SERVICE_WAIT"] = "300"
            #dataset = '_'.join(args.split.split('_')[:-1])
            wandb.init(project="NeighborMAE_EVAL", config=args, name=args.exp_name)

    trainer = Trainer(args)
    trainer.train()
    if args.wandb_log:
        wandb.finish()
