import os
import time

import torch

from dataset.music import MUSICMixDataset
from helpers.utils import AverageMeter, makedirs, get_ctx
from steps.common import build_model, get_underlying_nets, init_history, adjust_learning_rate
from steps.evaluate_base import _evaluate


def save_nets(ctx, suffix):
    path = get_ctx(ctx, 'path')
    nets = get_underlying_nets(get_ctx(ctx, 'net_wrapper'))
    (net_sound, net_frame, net_synthesizer) = nets

    torch.save(net_sound.state_dict(), os.path.join(path, f'sound_{suffix}'))
    torch.save(net_frame.state_dict(), os.path.join(path, f'frame_{suffix}'))
    torch.save(net_synthesizer.state_dict(), os.path.join(path, f'synthesizer_{suffix}'))


def checkpoint(ctx: dict):
    epoch = get_ctx(ctx, 'epoch')
    history = get_ctx(ctx, 'history')
    path = get_ctx(ctx, 'path')

    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(epoch, os.path.join(path, f'epoch_{suffix_latest}'))
    torch.save(history, os.path.join(path, f'history_{suffix_latest}'))
    save_nets(ctx, suffix_latest)

    cur_err = history['val']['err'][-1]
    if cur_err < get_ctx(ctx, 'best_err') and epoch % get_ctx(ctx, 'eval_epoch') == 0:
        ctx['best_err'] = cur_err
        save_nets(ctx, suffix_best)

    if get_ctx(ctx, 'checkpoint_epoch') is not None and epoch % get_ctx(ctx, 'checkpoint_epoch') == 0:
        save_nets(ctx, f'{epoch}.pth')


def synchronize(ctx: dict):
    if get_ctx(ctx, 'device').type != 'cpu':
        torch.cuda.synchronize()


def train_epoch(ctx: dict):
    net_wrapper = get_ctx(ctx, 'net_wrapper')
    optimizer = get_ctx(ctx, 'optimizer')
    loader = get_ctx(ctx, 'loader_train')
    history = get_ctx(ctx, 'history')
    epoch = get_ctx(ctx, 'epoch')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    net_wrapper.train()

    # main loop
    synchronize(ctx)
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        synchronize(ctx)
        data_time.update(time.perf_counter() - tic)

        # forward pass
        net_wrapper.zero_grad()
        err, _ = net_wrapper.forward(batch_data, ctx)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        synchronize(ctx)
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % get_ctx(ctx, 'disp_iter') == 0:
            print(f'Epoch: [{epoch}][{i}/{get_ctx(ctx, "epoch_iters")}],'
                  f' Time: {batch_time.average():.2f}, Data: {data_time.average():.2f}, '
                  f'lr_sound: {get_ctx(ctx, "lr_sound")}, lr_frame: {get_ctx(ctx, "lr_frame")}, '
                  f'lr_synthesizer: {get_ctx(ctx, "lr_synthesizer")}, '
                  f'loss: {err.item():.4f}')
            fractional_epoch = epoch - 1 + 1. * i / get_ctx(ctx, 'epoch_iters')
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def create_optimizer(nets, ctx):
    (net_sound, net_frame, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': get_ctx(ctx, 'lr_sound')},
                    {'params': net_synthesizer.parameters(), 'lr': get_ctx(ctx, 'lr_synthesizer')},
                    {'params': net_frame.features.parameters(), 'lr': get_ctx(ctx, 'lr_frame')},
                    {'params': net_frame.fc.parameters(), 'lr': get_ctx(ctx, 'lr_frame')}]
    return torch.optim.SGD(param_groups, momentum=get_ctx(ctx, 'beta1'),
                           weight_decay=get_ctx(ctx, 'weight_decay'))


def train(ctx: dict):
    ctx['net_wrapper'] = build_model(ctx)
    ctx['optimizer'] = create_optimizer(get_underlying_nets(get_ctx(ctx, 'net_wrapper')), ctx)

    dataset_train = MUSICMixDataset(get_ctx(ctx, 'list_train'), ctx, split='train')
    ctx['loader_train'] = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=get_ctx(ctx, 'batch_size'),
        shuffle=True,
        num_workers=int(get_ctx(ctx, 'workers')),
        drop_last=True)

    ctx['epoch_iters'] = len(dataset_train) // get_ctx(ctx, 'batch_size')
    print(f'1 Epoch = {get_ctx(ctx, "epoch_iters")} iters')

    dataset_val = MUSICMixDataset(get_ctx(ctx, 'list_val'), ctx,
                                  max_sample=get_ctx(ctx, 'num_val'), split='val')
    ctx['loader_val'] = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=get_ctx(ctx, 'batch_size'),
        shuffle=False,
        num_workers=2,
        drop_last=False)

    ctx['history'], from_epoch = init_history(ctx)
    if not get_ctx(ctx, 'continue_training'):
        makedirs(get_ctx(ctx, 'path'), remove=True)

    for epoch in range(from_epoch, get_ctx(ctx, 'num_epoch') + 1):
        ctx['epoch'] = epoch

        with torch.set_grad_enabled(True):
            train_epoch(ctx)

        with torch.set_grad_enabled(False):
            if epoch % get_ctx(ctx, 'eval_epoch') == 0:
                _evaluate(ctx)
            checkpoint(ctx)

        # drop learning rate
        if epoch in get_ctx(ctx, 'lr_steps'):
            adjust_learning_rate(ctx)

    print('Training Done!')
