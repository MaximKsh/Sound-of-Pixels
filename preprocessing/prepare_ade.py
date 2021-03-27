import argparse
import random
import os
import shutil
from pathlib import Path
import scipy.io.wavfile as wavfile
import numpy as np
from PIL import Image


def remkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def main(args):
    files = list(Path(args.scan_dir).rglob("*.jpg"))
    files = [f for f in files if 'room' in str(f)]
    random.shuffle(files)
    files = files[:args.samples]
    print(files)

    remkdir(args.out_dir)
    frames_dir = os.path.join(args.out_dir, 'frames')
    audio_dir = os.path.join(args.out_dir, 'audio')
    remkdir(frames_dir)
    remkdir(audio_dir)
    frames_dir = os.path.join(frames_dir, 'silence')
    audio_dir = os.path.join(audio_dir, 'silence')
    remkdir(frames_dir)
    remkdir(audio_dir)

    empty_audio = np.zeros((args.lensec * args.rate,))
    for i in range(args.samples):
        sample_id = f'{i:011x}'

        wavfile.write(os.path.join(audio_dir, sample_id + '.mp3'), args.rate, empty_audio)

        sample_frames_dir = os.path.join(frames_dir, sample_id + '.mp4')
        remkdir(sample_frames_dir)
        img = Image.open(files[i])
        img = img.resize((224, 224))
        img.save(os.path.join(sample_frames_dir, '000001.jpg'))

        for j in range(2, args.lensec * args.fps + 1):
            shutil.copyfile(os.path.join(sample_frames_dir, '000001.jpg'),
                            os.path.join(sample_frames_dir, f'{j:06}.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_dir', help='path to scene images')
    parser.add_argument('out_dir', help='Path to output audio/frames dir')
    parser.add_argument('--samples', help='How much silent samples we need', default=50, type=int)
    parser.add_argument('--lensec', default=60, type=int)
    parser.add_argument('--fps', default=8, type=int)
    parser.add_argument('--rate', default=11025, type=int)
    args = parser.parse_args()

    main(args)
