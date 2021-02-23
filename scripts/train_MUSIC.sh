#!/bin/bash

OPTS=""
OPTS+="--id MUSIC "
OPTS+="--list_train thesis/Sound-of-Pixels/data/train.csv "
OPTS+="--list_val thesis/Sound-of-Pixels/data/val.csv "
OPTS+="--ckpt thesis/Sound-of-Pixels/ckpt "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--output_activation sigmoid "
OPTS+="--img_activation tanh "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
OPTS+="--num_gpus 4 "
OPTS+="--workers 32 "
OPTS+="--batch_size_per_gpu 15 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 100 "
OPTS+="--eval_epoch 5 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "


# OPTS+="--dup_trainset 1 "
OPTS+="--seed 42 "

# OPTS+="--continue_training 1 "
OPTS+="--checkpoint_epoch 10 "
# OPTS+="--finetune ckpt/MUSIC-2mix-LogFreq-resnet18dilatedtanh-unet7no-linearsigmoid-frames3stride1-maxpool-binary-weightedLoss-channels64-epoch130-step50_100 "

echo $OPTS
python3 -u thesis/Sound-of-Pixels/main.py $OPTS
