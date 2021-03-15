from typing import Optional

import torch
import torch.nn as nn
from models.net_wrapper import ModelBuilder, NetWrapper


def build_model(context: dict):
    if context['load_best_model']:
        weights_sound = context['weights_sound_best']
        weights_frame = context['weights_frame_best']
        weights_synthesizer = context['weights_synthesizer_best']
    elif context['config']['continue_training']:
        weights_sound = context['weights_sound_latest']
        weights_frame = context['weights_frame_latest']
        weights_synthesizer = context['weights_synthesizer_latest']
    elif context['config']['finetune']:
        weights_sound = context['weights_sound_finetune']
        weights_frame = context['weights_frame_finetune']
        weights_synthesizer = context['weights_synthesizer_finetune']
    else:
        weights_sound, weights_frame, weights_synthesizer = '', '', ''

    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=context['config']['arch_sound'],
        fc_dim=context['config']['num_channels'],
        weights=weights_sound)
    net_frame = builder.build_frame(
        arch=context['config']['arch_frame'],
        fc_dim=context['config']['num_channels'],
        pool_type=context['config']['img_pool'],
        weights=weights_frame)
    net_synthesizer = builder.build_synthesizer(
        arch=context['config']['arch_synthesizer'],
        fc_dim=context['config']['num_channels'],
        weights=weights_synthesizer)

    if context['config']['finetune']:
        for param in net_sound.parameters():
            param.requires_grad = False
        for param in net_frame.parameters():
            param.requires_grad = False
    nets = (net_sound, net_frame, net_synthesizer)
    crit = builder.build_criterion(arch=context['config']['loss'])
    net_wrapper = NetWrapper(nets, crit)
    if context['device'].type != 'cpu':
        net_wrapper = nn.DataParallel(net_wrapper, device_ids=range(context['config']['num_gpus']))
        net_wrapper.to(context['device'])

    return net_wrapper


def get_underlying_nets(module: nn.Module):
    if isinstance(module, NetWrapper):
        return module.net_sound, module.net_frame, module.net_synthesizer
    if isinstance(module, nn.DataParallel):
        return module.module.net_sound, module.module.net_frame, module.module.net_synthesizer

    raise ValueError('module can be NetWrapper or nn.DataParallel')


def adjust_learning_rate(context):
    context['lr_sound'] *= 0.1
    context['lr_frame'] *= 0.1
    context['lr_synthesizer'] *= 0.1
    for param_group in context['optimizer'].param_groups:
        param_group['lr'] *= 0.1


def init_history(context: Optional[dict]):
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    if context and context['config']['continue_training']:
        suffix_latest = 'latest.pth'
        from_epoch = torch.load('{}/epoch_{}'.format(context['path'], suffix_latest)) + 1
        history = torch.load('{}/history_{}'.format(context, suffix_latest))

        for step in context['config']['lr_steps']:
            if step < from_epoch:
                adjust_learning_rate(context)
    else:
        from_epoch = 0

    return history, from_epoch
