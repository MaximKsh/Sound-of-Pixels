from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.utils import get_ctx, warpgrid
from models.net_wrapper import ModelBuilder, NetWrapper


def build_model(ctx: dict):
    if get_ctx(ctx, 'load_best_model'):
        weights_sound = get_ctx(ctx, 'weights_sound_best')
        weights_frame = get_ctx(ctx, 'weights_frame_best')
        weights_synthesizer = get_ctx(ctx, 'weights_synthesizer_best')
    elif get_ctx(ctx, 'continue_training') == 'latest':
        weights_sound = get_ctx(ctx, 'weights_sound_latest')
        weights_frame = get_ctx(ctx, 'weights_frame_latest')
        weights_synthesizer = get_ctx(ctx, 'weights_synthesizer_latest')
    elif isinstance(get_ctx(ctx, 'continue_training'), int):
        ep = get_ctx(ctx, 'continue_training')
        weights_sound = get_ctx(ctx, f'weights_sound_{ep}')
        weights_frame = get_ctx(ctx, f'weights_frame_{ep}')
        weights_synthesizer = get_ctx(ctx, f'weights_synthesizer_{ep}')
    elif get_ctx(ctx, 'finetune'):
        weights_sound = get_ctx(ctx, 'weights_sound_finetune')
        weights_frame = get_ctx(ctx, 'weights_frame_finetune')
        weights_synthesizer = get_ctx(ctx, 'weights_synthesizer_finetune')
    else:
        weights_sound, weights_frame, weights_synthesizer = '', '', ''

    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=get_ctx(ctx, 'arch_sound'),
        fc_dim=get_ctx(ctx, 'num_channels'),
        weights=weights_sound)
    net_frame = builder.build_frame(
        arch=get_ctx(ctx, 'arch_frame'),
        fc_dim=get_ctx(ctx, 'num_channels'),
        pool_type=get_ctx(ctx, 'img_pool'),
        weights=weights_frame)
    net_synthesizer = builder.build_synthesizer(
        arch=get_ctx(ctx, 'arch_synthesizer'),
        fc_dim=get_ctx(ctx, 'num_channels'),
        weights=weights_synthesizer)

    if get_ctx(ctx, 'finetune'):
        for param in net_sound.parameters():
            param.requires_grad = False
        for param in net_frame.parameters():
            param.requires_grad = False
    nets = (net_sound, net_frame, net_synthesizer)
    crit = builder.build_criterion(arch=get_ctx(ctx, 'loss'))
    net_wrapper = NetWrapper(nets, crit)
    if get_ctx(ctx, 'device').type != 'cpu':
        net_wrapper = nn.DataParallel(net_wrapper, device_ids=get_ctx(ctx, 'gpu'))
        # net_wrapper.to(get_ctx(ctx, 'device'))

    return net_wrapper


def get_underlying_nets(module: nn.Module):
    if isinstance(module, NetWrapper):
        return module.net_sound, module.net_frame, module.net_synthesizer
    if isinstance(module, nn.DataParallel):
        return module.module.net_sound, module.module.net_frame, module.module.net_synthesizer

    raise ValueError('module can be NetWrapper or nn.DataParallel')


def adjust_learning_rate(ctx):
    ctx['lr_sound'] *= 0.1
    ctx['lr_frame'] *= 0.1
    ctx['lr_synthesizer'] *= 0.1
    for param_group in get_ctx(ctx, 'optimizer').param_groups:
        param_group['lr'] *= 0.1


def init_history(ctx: Optional[dict]):
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    continue_training = get_ctx(ctx, 'continue_training')
    if ctx and continue_training == 'latest' or isinstance(continue_training, int):
        suffix_latest = 'latest.pth'
        from_epoch = torch.load('{}/epoch_{}'.format(get_ctx(ctx, 'path'), suffix_latest)) + 1
        history = torch.load('{}/history_{}'.format(get_ctx(ctx, 'path'), suffix_latest))

        if isinstance(continue_training, int):
            from_epoch = get_ctx(ctx, 'continue_training')
            for k in history:
                for k1 in history[k]:
                    history[k][k1] = history[k][k1][:from_epoch]

        for step in get_ctx(ctx, 'lr_steps'):
            if step < from_epoch:
                adjust_learning_rate(ctx)
    else:
        from_epoch = 0

    return history, from_epoch


def unwarp_log_scale(ctx, arr):
    N = get_ctx(ctx, 'num_mix')
    B = arr[0].size(0)
    linear = [None for _ in range(N)]

    for n in range(N):
        if get_ctx(ctx, 'log_freq'):
            w = warpgrid(B, get_ctx(ctx, 'stft_frame') // 2 + 1, arr[0].size(3), warp=False)
            grid_unwarp = torch.from_numpy(w).to(get_ctx(ctx, 'device'))
            linear[n] = F.grid_sample(arr[n], grid_unwarp, align_corners=True)
        else:
            linear[n] = arr[n]

    return linear


def detach_mask(ctx, mask, binary):
    N = get_ctx(ctx, 'num_mix')
    for n in range(N):
        mask[n] = mask[n].detach().cpu().numpy()
        if binary:
            mask[n] = (mask[n] > get_ctx(ctx, 'mask_thres')).astype(np.float32)

    return mask


def to_device(ctx, data):
    if get_ctx(ctx, 'device').type != 'cpu' and len(get_ctx(ctx, 'gpu')) == 1:
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(get_ctx(ctx, 'device'))
