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
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--test_ratio', default=0.05, type=float)
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext='.mp3')
    for audio_path in audio_files:
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace('.mp3', '.mp4')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) > args.fps * 20:
            infos.append((audio_path, frame_path, str(len(frame_files))))
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_val = int(len(infos) * args.val_ratio)
    n_test = int(len(infos) * args.test_ratio) 
    random.shuffle(infos)
    
    valset_indexes = set()
    testset_indexes = set()
    i = 0
    while len(valset_indexes) < n_val or len(testset_indexes) < n_test:
        audio_path = infos[i][0]
        instrument = os.path.split(os.path.split(audio_path)[0])[1]
        if ' ' in instrument and len(testset_indexes) < n_test:
            testset_indexes.add(i)
        if not ' ' in instrument and instrument != 'silence' and len(valset_indexes) < n_val:
            valset_indexes.add(i)
        i += 1
        
    infos = np.array([','.join(x) for x in infos])
        
    testset = infos[list(testset_indexes)]
    valset = infos[list(valset_indexes)]
    trainset = infos[list(set(range(len(infos))) - valset_indexes - testset_indexes)]
    
    # trainset = infos[0:n_train]
    # valset = infos[n_train:]
    for name, subset in zip(['train', 'val', 'test'], [trainset, valset, testset]):
        filename = '{}.csv'.format(os.path.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')
