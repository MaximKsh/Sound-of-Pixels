from typing import List
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset.music import MUSICMixDataset
from helpers.utils import get_ctx
from steps.common import build_model, get_underlying_nets


def set_lr(opimizer, lr, groups):
    for g in groups:
        opimizer.param_groups[g]['lr'] = lr


def create_optimizer(nets, ctx):
    (net_sound, net_frame, net_synthesizer) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': get_ctx(ctx, 'lr_sound')},
                    {'params': net_synthesizer.parameters(), 'lr': get_ctx(ctx, 'lr_synthesizer')},
                    {'params': net_frame.features.parameters(), 'lr': get_ctx(ctx, 'lr_frame')},
                    {'params': net_frame.fc.parameters(), 'lr': get_ctx(ctx, 'lr_frame')}]
    return torch.optim.SGD(param_groups, momentum=get_ctx(ctx, 'beta1'),
                           weight_decay=get_ctx(ctx, 'weight_decay'))


def lr_range_test_for_part(ctx: dict, groups: List[int]):
    ctx['net_wrapper'] = build_model(ctx)
    ctx['optimizer'] = create_optimizer(get_underlying_nets(get_ctx(ctx, 'net_wrapper')), ctx)
    dataset_train = MUSICMixDataset(get_ctx(ctx, 'list_train'), ctx, split='train')
    ctx['loader_train'] = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=get_ctx(ctx, 'batch_size'),
        shuffle=True,
        num_workers=int(get_ctx(ctx, 'workers')),
        drop_last=True)

    rates = []
    losses = []

    net_wrapper = get_ctx(ctx, 'net_wrapper')
    optimizer = get_ctx(ctx, 'optimizer')
    loader = get_ctx(ctx, 'loader_train')

    total = 1000
    min_lr = 1e-10
    max_lr = 1e1
    smooth_f = 0.05
    for step, batch_data in tqdm(enumerate(loader), total=total):
        if step == total:
            break

        it = step / total
        lr = np.exp((1 - it) * np.log(min_lr) + it * np.log(max_lr))

        set_lr(optimizer, lr, groups)

        net_wrapper.zero_grad()
        loss, _ = net_wrapper.forward(batch_data, ctx)
        loss = loss.mean()

        # backward
        loss.backward()
        optimizer.step()

        if step > 0:
            loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
        rates.append(lr)
        losses.append(loss.item())

    return np.array(rates), np.array(losses)


def lr_range_test(ctx: dict):
    plt.figure()

    print(f'Audio net')
    rates, losses = lr_range_test_for_part(ctx, [0])
    best = np.argmin(losses)
    print(f'Best loss: {losses[best]}, lr: {rates[best]}')
    print(f'Suggest: {rates[best] / 10}')

    plt.subplot(3, 1, 1)
    plt.xscale('log')
    plt.title('Audio')
    q = np.quantile(rates, 0.9)
    idx = np.argwhere(rates < q)
    plt.plot(rates[idx], losses[idx])

    print(f'Synthesizer net')
    rates, losses = lr_range_test_for_part(ctx, [1])
    best = np.argmin(losses)
    print(f'Best loss: {losses[best]}, lr: {rates[best]}')
    print(f'Suggest: {rates[best] / 10}')

    plt.subplot(3, 1, 2)
    plt.xscale('log')
    plt.title('Synthesizer')
    q = np.quantile(rates, 0.9)
    idx = np.argwhere(rates < q)
    plt.plot(rates[idx], losses[idx])

    print(f'Frame net')
    rates, losses = lr_range_test_for_part(ctx, [2, 3])
    best = np.argmin(losses)
    print(f'Best loss: {losses[best]}, lr: {rates[best]}')
    print(f'Suggest: {rates[best] / 10}')

    plt.subplot(3, 1, 3)
    plt.xscale('log')
    plt.title('Frame')
    q = np.quantile(rates, 0.9)
    idx = np.argwhere(rates < q)
    plt.plot(rates[idx], losses[idx])

    plt.savefig('lr_rate_test.png')
