import torch

from dataset.music import MUSICMixDataset
from steps.common import build_model, init_history
from steps.evaluate_base import _evaluate


def evaluate(context: dict):
    context['load_best_model'] = True
    context['net_wrapper'] = build_model(context)

    dataset_val = MUSICMixDataset(context['config']['list_val'], context['config'],
                                  max_sample=context['config']['num_val'], split='val')
    context['loader_val'] = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=context['batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=False)

    context['history'], _ = init_history(None)

    context['epoch'] = 0
    with torch.set_grad_enabled(False):
        _evaluate(context)
