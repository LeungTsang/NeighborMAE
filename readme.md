This is the code repository for our CVPR 2026 Paper: 


## 🌍 NeighborMAE: Exploiting Spatial Dependencies between Neighboring Earth Observation Images in Masked Autoencoders

Liang Zeng, Valerio Marsocci, Wufan Zhao, Andrea Nascetti, Maarten Vergauwen

### 🚀 Overview

NeighborMAE is a novel self-supervised learning framework for Earth Observation (EO) imagery that extends Masked Autoencoders (MAE) by explicitly modeling spatial dependencies between neighboring images.


### Models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">checkpoints</td>
<td align="center"><a href="https://drive.google.com/file/d/1YIun61QPHraCO-ro2skRyPPDoEnwYhaz/view?usp=sharing">download</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1t_ys1QwCcJZ7fmLslbRTGlV_6aaVKLYk/view?usp=sharing">download</a></td>
</tr>
</tbody></table>


### 📦 Prerequisites
```
Pytorch >= 2.0
Nvidia-DALI >= 1.49.0 (optional to accelerate data loading)
```


### 📁 Dataset Preparation (fMoW-RGB)

We use the train set of the raw fMoW-RGB dataset for pre-training, with the below directory structure:

```
data/
  fmow/
    train/
      airport/
        airport_0/
          airport_0_0_rgb.jpg
          airport_0_1_rgb.jpg
          ...
        airport_1/
        ...
      airport_hangar/
        ...
      ...
      zoo/
        ...
```

For evaulation, we follow the preprocessing of fMoW baseilne (https://github.com/fMoW/baseline), which crops the bounded area out for each image. The directory structure remains similar: 

```
data/
  fmow_baseline/
    train/
      airport/
        airport_0/
          airport_0_0_rgb.jpg
          airport_0_1_rgb.jpg
          ...
        airport_1/
        ...
      airport_hangar/
        ...
      ...
      zoo/
        ...
	val/
	  ...
```


### 🏋️ Pretraining

To train ViT-Large on fMoW for 800 epochs with a total batch size of 2048 on a 4-GPU node: 


```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --rdzv_endpoint=localhost:29500 \
run.py --seed <seed> --exp_name <experiment_name> --wandb_log \
--output_path <output_path> --data_path <path_to_fMoW> --dali \
--base_lr 1.5e-4 --min_lr 0 --epochs 800 --batch_size 512 --warmup_ratio 0.05 \
--img_size 224 --patch_size 16 --in_chans 3 --aspect_ratio 0.75 --crop_scale 0.2 1.0 \
--mask_ratio_min 0.75 --mask_ratio_max 0.85 \
--norm_pix_loss --filter_norm_bias --fp16 \
--save_interval 100 --log_interval 10 \
--embed_dim 1024 --depth 24 --num_heads 16 


```


### 🔍 Evaluation

Convert checkpoint to standard ViT (timm) weight:
```
python tools/convert_vit.pth <checkpoint_path> <output_weight_path>
```

#### Linear probing
```

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --rdzv_endpoint=localhost:29500 \
eval_cls/eval_cls.py --seed <seed> --exp_name <experiment_name> --wandb_log \
--resume_path <path_to_weight> --load_model_only --output_path <output_path> --dali \
--save_interval 10 --log_interval 10 --eval_interval 5 \
--data_path <path_to_fMoW_baseline>  --classes_num 62 \
--split split/fmow_baseline_train.txt --val_split split/fmow_baseline_val.txt \
--base_lr 1e-3 --min_lr 1e-6 --epochs 20 --batch_size 128 \
--img_size 224 --patch_size 16 --in_chans 3 --aspect_ratio 0.75 --crop_scale 0.2 1.0 \
--wd 0.00 --betas 0.9 0.999 --fp16  --warmup_ratio 0.05 \
--global_pool token --drop_path 0.0 --freeze_backbone \
--embed_dim 1024 --depth 24 --num_heads 16 

```

#### Fine-tuning

```
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --rdzv_endpoint=localhost:29500 \
eval_cls/eval_cls.py --seed <seed> --exp_name <experiment_name> --wandb_log \
--resume_path <path_to_weight> --load_model_only --output_path <output_path> --dali \
--save_interval 10 --log_interval 10 --eval_interval 5 \
--data_path <path_to_fMoW_baseline>  --classes_num 62 \
--split split/fmow_baseline_train.txt --val_split split/fmow_baseline_val.txt \
--base_lr 1e-3 --min_lr 1e-6 --epochs 20 --batch_size 128 \
--img_size 224 --patch_size 16 --in_chans 3 --aspect_ratio 0.75 --crop_scale 0.2 1.0 \
--wd 0.05 --betas 0.9 0.999 --fp16  --warmup_ratio 0.05 \
--global_pool avg --mixup 0.8 --cutmix 1.0 --drop_path 0.2 --layer_decay 0.75 \
--embed_dim 1024 --depth 24 --num_heads 16

```


