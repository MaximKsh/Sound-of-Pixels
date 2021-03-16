import os

import torch
from tqdm import tqdm

from dataset.music import MUSICMixDataset
from helpers.utils import get_ctx, makedirs
from helpers.viz import HTMLVisualizer
from steps.common import build_model


def output_visuals(vis_rows, batch_data, outputs, ctx):
    raise NotImplementedError


def _test(ctx: dict):
    makedirs(get_ctx(ctx, 'vis_test'), remove=True)

    net_wrapper = get_ctx(ctx, 'net_wrapper')
    net_wrapper.eval()
    loader = get_ctx(ctx, 'loader_test')

    # initialize HTML header
    visualizer = HTMLVisualizer(os.path.join(get_ctx(ctx, 'vis_test'), 'index.html'))
    header = ['Filename', 'Input Audio']
    for n in range(1, get_ctx(ctx, 'num_mix') + 1):
        header += [f'Predicted Audio {n:d}', f'Predicted Mask {n}']
    visualizer.add_header(header)
    vis_rows = []

    for i, batch_data in tqdm(enumerate(loader)):
        _, outputs = net_wrapper.forward(batch_data, ctx)

        if len(vis_rows) < get_ctx(ctx, 'num_vis'):
            output_visuals(vis_rows, batch_data, outputs, ctx)

    visualizer.add_rows(vis_rows)
    visualizer.write_html()


def test(ctx: dict):
    ctx['load_best_model'] = True
    ctx['net_wrapper'] = build_model(ctx)
    ctx['num_mix'] = 1

    dataset = MUSICMixDataset(get_ctx(ctx, 'list_test'), ctx, max_sample=get_ctx(ctx, 'num_test'), split='test')
    ctx['loader_test'] = torch.utils.data.DataLoader(dataset, batch_size=get_ctx(ctx, 'batch_size'),
                                                     shuffle=False, num_workers=2, drop_last=False)

    with torch.set_grad_enabled(False):
        _test(ctx)
