import torch

from dataset.music import MUSICMixDataset
from helpers.utils import get_ctx
from steps.common import build_model, init_history
from steps.evaluate_base import _evaluate


def evaluate(ctx: dict):
    ctx['load_best_model'] = True
    ctx['net_wrapper'] = build_model(ctx)

    dataset_val = MUSICMixDataset(get_ctx(ctx, 'list_val'), ctx, max_sample=get_ctx(ctx, 'num_val'), split='val')
    ctx['loader_val'] = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=get_ctx(ctx, 'batch_size'),
        shuffle=False,
        num_workers=2,
        drop_last=False)

    ctx['history'], _ = init_history(None)

    ctx['epoch'] = 0
    with torch.set_grad_enabled(False):
        _evaluate(ctx)
