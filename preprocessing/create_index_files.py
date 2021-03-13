import os
import glob
import argparse
import random
import fnmatch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_root', help='Path to extracted audio files (<videoID>.mp3) and '
                                           'frames (xxxxxx.jpg frames inside <videoID> subdirs).')
    parser.add_argument('data_key', help='Data related short identifier. It will used as data directory suffix.'
                                         'Files will be saved inside ./data-{data-key}.')
    parser.add_argument('--min_frames', default=160, type=int, help='Minimal number of frames required to pair matching'
                                                                    'If number of frames inside <videoID> subdir less'
                                                                    ' than min_frames then pair will be ignored')
    parser.add_argument('--val_ratio', default=0.15, type=float, help='Ratio of validation subset')
    parser.add_argument('--test_ratio', default=0.1, type=float, help='Ratio of test subset')
    parser.add_argument('--check_duets', default=1, type=int, help='If true test set must contains only duets and'
                                                                   ' val set must contains only solos')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    root_audio = os.path.join(args.input_root, 'audio')
    root_frame = os.path.join(args.input_root, 'frames')
    output_path = f'data-{args.data_key}'

    print('Audio root:', root_audio)
    print('Frames root:', root_frame)
    print('Output path:', output_path)

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(root_audio, ext='.mp3')
    print('Audio files found: ', len(audio_files))
    print('Matching frames...')
    for audio_path in tqdm(audio_files):
        frame_path = audio_path.replace(root_audio, root_frame) \
            .replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if args.min_frames == 0 or len(frame_files) > args.min_frames:
            infos.append((f'"{audio_path}"', f'"{frame_path}"', str(len(frame_files))))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_val = int(len(infos) * args.val_ratio)
    n_test = int(len(infos) * args.test_ratio)
    random.shuffle(infos)

    valset_indexes = set()
    testset_indexes = set()
    i = 0
    print('Train/val/test splitting...')
    while i < len(infos) and (len(valset_indexes) < n_val or len(testset_indexes) < n_test):
        audio_path = infos[i][0]
        instrument = os.path.split(os.path.split(audio_path)[0])[1]
        if (not args.check_duets or ' ' in instrument) and len(testset_indexes) < n_test:
            testset_indexes.add(i)
        if (not args.check_duets or ' ' not in instrument) and instrument != 'silence' and len(valset_indexes) < n_val:
            valset_indexes.add(i)
        i += 1

    infos = np.array([','.join(x) for x in infos])

    testset = infos[list(testset_indexes)]
    valset = infos[list(valset_indexes)]
    trainset = infos[list(set(range(len(infos))) - valset_indexes - testset_indexes)]

    random.shuffle(testset)
    random.shuffle(valset)
    random.shuffle(trainset)
    print(f'Trainset size: {len(trainset)}, Valset size: {len(valset)}, Testset size: {len(testset)}')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print('Writing output files...')
    for name, subset in zip(['train', 'val', 'test'], [trainset, valset, testset]):
        filename = '{}.csv'.format(os.path.join(output_path, name))
        print(f'Writing to {filename}...')
        with open(filename, 'w') as f:
            for item in tqdm(subset):
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
