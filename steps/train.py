import os
import time

import torch

from dataset.music import MUSICMixDataset
from helpers.utils import AverageMeter, makedirs
from steps.common import build_model, get_underlying_nets, init_history, adjust_learning_rate
from steps.evaluate_base import _evaluate


def save_nets(context, suffix):
    path = context['path']
    nets = get_underlying_nets(context['net_wrapper'])
    (net_sound, net_frame, net_synthesizer) = nets

    torch.save(net_sound.state_dict(), os.path.join(path, f'sound_{suffix}'))
    torch.save(net_frame.state_dict(), os.path.join(path, f'frame_{suffix}'))
    torch.save(net_synthesizer.state_dict(), os.path.join(path,f'synthesizer_{suffix}'))


def checkpoint(context: dict):
    epoch = context['epoch']
    history = context['history']
    path = context['path']

    print('Saving checkpoints at {} epochs.'.format(epoch))
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(epoch, os.path.join(path, f'epoch_{suffix_latest}'))
    torch.save(history, os.path.join(path, f'history_{suffix_latest}'))
    save_nets(context, suffix_latest)

    cur_err = history['val']['err'][-1]
    if cur_err < context['best_err'] and epoch % context['config']['eval_epoch'] == 0:
        context['best_err'] = cur_err
        save_nets(context, suffix_best)

    if context['config']['checkpoint_epoch'] is not None and epoch % context['config']['checkpoint_epoch'] == 0:
        save_nets(context, f'{epoch}.pth')


def synchronize(context: dict):
    if context['device'].type != 'cpu':
        torch.cuda.synchronize()


def train_epoch(context: dict):
    net_wrapper = context['net_wrapper']
    optimizer = context['optimizer']
    loader = context['loader_train']
    history = context['history']
    epoch = context['epoch']

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    net_wrapper.train()

    # main loop
    synchronize(context)
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        synchronize(context)
        data_time.update(time.perf_counter() - tic)

        # forward pass
        net_wrapper.zero_grad()
        err, _ = net_wrapper.forward(batch_data, context)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        synchronize(context)
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % context['config']['disp_iter'] == 0:
            print(f'Epoch: [{epoch}][{i}/{context["epoch_iters"]}],'
                  f' Time: {batch_time.average():.2f}, Data: {data_time.average():.2f}, '
                  f'lr_sound: {context["lr_sound"]}, lr_frame: {context["lr_frame"]}, '
                  f'lr_synthesizer: {context["lr_synthesizer"]}, '
                  f'loss: {err.item():.4f}')
            fractional_epoch = epoch - 1 + 1. * i / context['epoch_iters']
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def create_optimizer(nets, context):
    (net_sound, net_frame, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': context['lr_sound']},
                    {'params': net_synthesizer.parameters(), 'lr': context['lr_synthesizer']},
                    {'params': net_frame.features.parameters(), 'lr': context['lr_frame']},
                    {'params': net_frame.fc.parameters(), 'lr': context['lr_frame']}]
    return torch.optim.SGD(param_groups, momentum=context['config']['beta1'],
                           weight_decay=context['config']['weight_decay'])


def train(context: dict):
    context['net_wrapper'] = build_model(context)
    context['optimizer'] = create_optimizer(get_underlying_nets(context['net_wrapper']), context)

    dataset_train = MUSICMixDataset(context['config']['list_train'], context['config'], split='train')
    context['loader_train'] = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=context['batch_size'],
        shuffle=True,
        num_workers=int(context['config']['workers']),
        drop_last=True)

    context['epoch_iters'] = len(dataset_train) // context['batch_size']
    print('1 Epoch = {} iters'.format(context['epoch_iters']))

    dataset_val = MUSICMixDataset(context['config']['list_val'], context['config'],
                                  max_sample=context['config']['num_val'], split='val')
    context['loader_val'] = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=context['batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=False)

    context['history'], from_epoch = init_history(context)
    if not context['config']['continue_training']:
        makedirs(context['path'], remove=True)

    for epoch in range(from_epoch, context['config']['num_epoch'] + 1):
        context['epoch'] = epoch

        with torch.set_grad_enabled(True):
            train_epoch(context)

        with torch.set_grad_enabled(False):
            if epoch % context['config']['eval_epoch'] == 0:
                _evaluate(context)
            checkpoint(context)

        # drop learning rate
        if epoch in context['config']['lr_steps']:
            adjust_learning_rate(context)

    print('Training Done!')
