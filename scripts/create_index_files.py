import os
import glob
import argparse
import random
import fnmatch
import numpy as np

def find_recursive(root_dir, ext='.mp3'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


def find_pairs(audio_files, root_audio, root_frame, fps=8):
    infos = []
    for audio_path in audio_files:
        frame_path = audio_path.replace(root_audio, root_frame) \
                               .replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if fps == -1 or len(frame_files) > fps * 20:
            infos.append((f'"{audio_path}"', f'"{frame_path}"', str(len(frame_files))))
            
    return infos


def train_val_test_split(infos, n_val, n_test, check_duets=True):
    valset_indexes = set()
    testset_indexes = set()
    i = 0
    while i < len(infos) and (len(valset_indexes) < n_val or len(testset_indexes) < n_test):
        audio_path = infos[i][0]
        instrument = os.path.split(os.path.split(audio_path)[0])[1]
        if (not check_duets or ' ' in instrument) and len(testset_indexes) < n_test:
            testset_indexes.add(i)
        if (not check_duets or not ' ' in instrument) and instrument != 'silence' and len(valset_indexes) < n_val:
            valset_indexes.add(i)
        i += 1
    return valset_indexes, testset_indexes


def write_sets(trainset, valset, testset, path_output):
    for name, subset in zip(['train', 'val', 'test'], [trainset, valset, testset]):
        filename = '{}.csv'.format(os.path.join(path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/audio',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/frames',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data',
                        help="path to output index files")
    parser.add_argument('--val_ratio', default=0.15, type=float)
    parser.add_argument('--test_ratio', default=0.1, type=float)
    parser.add_argument('--check_duets', default=1, type=int)
    args = parser.parse_args()

    # find all audio/frames pairs
    
    audio_files = find_recursive(args.root_audio, ext='.mp3')
    infos = find_pairs(audio_files, args.root_audio, args.root_frame, args.fps)
    
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/v
    n_val = int(len(infos) * args.val_ratio)
    n_test = int(len(infos) * args.test_ratio)
    random.shuffle(infos)
    
    valset_indexes, testset_indexes = train_val_test_split(infos, n_val, n_test, args.check_duets)
    infos = np.array([','.join(x) for x in infos])
        
    testset = infos[list(testset_indexes)]
    valset = infos[list(valset_indexes)]
    trainset = infos[list(set(range(len(infos))) - valset_indexes - testset_indexes)]
    
    random.shuffle(testset)
    random.shuffle(valset)
    random.shuffle(trainset)
    
    write_sets(trainset, valset, testset, args.path_output)

    print('Done!')
