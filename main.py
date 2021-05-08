import argparse
import random
import numpy as np
import torch

from helpers.utils import read_config, create_context, get_ctx
from steps.test import test
from steps.evaluate import evaluate
from steps.train import train
from steps.regions import regions
from steps.lr_range_test import lr_range_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='train/eval/regions/lr_range_test')
    parser.add_argument('--config', default='', help='Configuration in JSON file')
    args = parser.parse_args()

    config = read_config('configs/default.json')
    if args.config != '':
        config_override = read_config(args.config)
        config = {**config, **config_override}

    ctx = create_context(config)

    random.seed(get_ctx(ctx, 'seed'))
    np.random.seed(get_ctx(ctx, 'seed'))
    torch.manual_seed(get_ctx(ctx, 'seed'))

    if args.mode == 'train':
        train(ctx)
    elif args.mode == 'eval':
        evaluate(ctx)
    # elif args.mode == 'test':
    #     test(ctx)
    elif args.mode == 'regions':
        regions(ctx)
    elif args.mode == 'lr_range_test':
        lr_range_test(ctx)
    else:
        raise RuntimeError(f'Unsupported mode {args.mode}, please use train/eval/regions/lr_range_test')
