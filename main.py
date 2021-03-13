import argparse
import random
import numpy as np
import torch

from helpers.utils import read_config, create_context
from steps.test import test
from steps.evaluate import evaluate
from steps.train import train
from steps.regions import regions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='train/eval/test/regions')
    parser.add_argument('--config', default='', help='Configuration in JSON file')
    args = parser.parse_args()

    config = read_config('configs/default.json')
    if args.config != '':
        config_override = read_config(args.config)
        config = {**config, **config_override}

    context = create_context(config)

    random.seed(context['config']['seed'])
    np.random.seed(context['config']['seed'])
    torch.manual_seed(context['config']['seed'])

    if args.mode == 'train':
        train(context)
    elif args.mode == 'eval':
        evaluate(context)
    elif args.mode == 'test':
        test(context)
    elif args.mode == 'regions':
        regions(context)
    else:
        raise RuntimeError(f'Unsupported mode {args.mode}, please use train/eval/test/regions')
