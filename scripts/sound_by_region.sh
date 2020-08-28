#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id MUSIC-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride24-maxpool-binary-weightedLoss-channels32-epoch10000-step4000_8000 "
OPTS+="--list_val data/test.csv "

# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--mask_thres 0.6 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 1 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

python3 -u sound_by_region.py $OPTS
