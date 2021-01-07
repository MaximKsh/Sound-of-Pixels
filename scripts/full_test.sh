#!/bin/bash

COMMON=""
COMMON+="--mode eval "
COMMON+="--id MUSIC-2mix-LogFreq-resnet18dilatedtanh-unet7no-linearsigmoid-frames3stride24-maxpool-binary-weightedLoss-channels32-epoch100-step40_80 "
# Models
COMMON+="--arch_sound unet7 "
COMMON+="--arch_synthesizer linear "
COMMON+="--arch_frame resnet18dilated "
COMMON+="--img_pool maxpool "
COMMON+="--num_channels 32 "
COMMON+="--img_activation tanh "
# binary mask, BCE loss, weighted loss
COMMON+="--binary_mask 1 "
COMMON+="--loss bce "
COMMON+="--weighted_loss 1 "
# logscale in frequency
COMMON+="--num_mix 2 "
COMMON+="--log_freq 1 "

# frames-related
COMMON+="--num_frames 3 "
COMMON+="--stride_frames 24 "
COMMON+="--frameRate 8 "

# audio-related
COMMON+="--audLen 65535 "
COMMON+="--audRate 11025 "




# Evaluate validation
OPTS=""
OPTS+=$COMMON
OPTS+="--num_vis 50 "
OPTS+="--num_val 50 "
OPTS+="--list_val data/val.csv "
OPTS+="--vis_dir val_evaluation "

python3 -u main.py $OPTS

# Evaluate test
OPTS=""
OPTS+=$COMMON
OPTS+="--num_vis 50 "
OPTS+="--num_val 50 "
OPTS+="--list_val data/test.csv "
OPTS+="--vis_dir test_evaluation "

python3 -u main.py $OPTS

# SBR&RBS val
OPTS=""
OPTS+=$COMMON
OPTS+="--list_val data/val.csv "
OPTS+="--vis_dir val_sbr&rbs "
python3 -u sound_by_region.py $OPTS

# SBR&RBS val
OPTS=""
OPTS+=$COMMON
OPTS+="--list_val data/test.csv "
OPTS+="--vis_dir test_sbr&rbs "
python3 -u sound_by_region.py $OPTS
