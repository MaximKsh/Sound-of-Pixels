import argparse
import os
from os import path as p
import shutil
import subprocess


# ffmpeg -i "BhXj3rVWGZ0.mp4" -ar 11025 -ac 1 -f mp3 abc.mp3
def extract_audio_ffmpeg(input_video, output_filename):
    try:
        subprocess.check_call([
            'ffmpeg', '-loglevel', 'warning','-i', input_video,
            '-ar', '11025', '-ac', '1', '-f', 'mp3', 
            output_filename])
    except Exception as ex:
        print(ex)


def ensure_empty_dir(args, required_dir):
    path = p.join(args.data_path, required_dir)
    if p.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return path


def extract_audio(args):
    print('Preparing audio dir...')
    audio_path = ensure_empty_dir(args, 'audio')

    print('Extract audio')
    for root, _, files in os.walk(args.raw_path):
        for name in files:
            clean_name, ext = p.splitext(name)
            if ext == '.mp4':
                _, instrument = p.split(root)
                print(instrument, name)
                
                instrument_dir = p.join(audio_path, instrument)

                if not p.exists(instrument_dir):
                    os.mkdir(instrument_dir)
                
                output_filename = p.join(instrument_dir, clean_name) + '.mp3'
                extract_audio_ffmpeg(p.join(root, name), output_filename)


# ffmpeg -i "BhXj3rVWGZ0.mp4" -r 8  frames/out-%03d.jpg
def extract_frames_ffmpeg(input_video, output_frames_path):
    try:
        subprocess.check_call([
            'ffmpeg', '-loglevel', 'error','-i', input_video, '-s', '224x224',
            '-r', '8', p.join(output_frames_path, '%06d.jpg')])
    except Exception as ex:
        print(ex)


def extract_frames(args):
    print('Preparing frames dir')
    frames_path = ensure_empty_dir(args, 'frames')

    print('Extract frames')
    for root, _, files in os.walk(args.raw_path):
        for name in files:
            _, ext = p.splitext(name)
            if ext == '.mp4':
                _, instrument = p.split(root)
                instrument_dir = p.join(frames_path, instrument)
                print(instrument, name)

                if not p.exists(instrument_dir):
                    os.mkdir(instrument_dir)
                
                output_frames_path = p.join(instrument_dir, name)
                os.mkdir(output_frames_path)
                extract_frames_ffmpeg(p.join(root, name), output_frames_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', help='Path to raw videos (input) directory')
    parser.add_argument('data_path', help='Path to data (output) directory')
    args = parser.parse_args()

    extract_frames(args)
    extract_audio(args)
