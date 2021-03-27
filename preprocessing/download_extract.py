import glob
import logging
import datetime
import random
import subprocess
import os
import argparse
import shutil
import time
from functools import partial
from typing import NamedTuple
import csv
import json
import multiprocessing as mp
import tqdm
import youtube_dl

logger = logging.getLogger()

DownloadItem = NamedTuple('DownloadItem', [
    ('youtube_id', str),
    ('label', str),
    ('start_seconds', int),
    ('len_sec', int)
])


def download_video(videos, output_path):
    ydl_opts = {
        'outtmpl': os.path.join(output_path, '%(id)s.%(ext)s'),
        'quiet': True,
        'format': '(mp4)[height<=360]',
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(videos)


def safe_subprocess_call(args):
    try:
        return_error = subprocess.check_call(args)
    except:
        cmd = ' '.join(args)
        logger.exception(cmd)
        return False

    if return_error != 0:
        cmd = ' '.join(args)
        logger.error('%s return non zero code %d', cmd, return_error)
        return False

    return True


# ffmpeg -i "BhXj3rVWGZ0.mp4" -ar 11025 -ac 1 -f mp3 abc.mp3
def extract_audio_ffmpeg(input_video, output_filename, start_sec=-1, len_sec=-1):
    args = ['ffmpeg', '-loglevel', 'error']
    if start_sec >= 0:
        ss = str(datetime.timedelta(seconds=start_sec))
        args += ['-ss', ss]
    args += ['-i', input_video]
    if len_sec >= 0:
        to = str(datetime.timedelta(seconds=len_sec))
        args += ['-to', to]

    args += ['-ar', '11025', '-ac', '1', '-f', 'mp3', output_filename]
    return safe_subprocess_call(args)


# ffmpeg -i "BhXj3rVWGZ0.mp4" -r 8  frames/out-%03d.jpg
def extract_frames_ffmpeg(input_video, output_frames_path, fps, start_sec=-1, len_sec=-1):
    args = ['ffmpeg', '-loglevel', 'error']
    if start_sec >= 0:
        ss = str(datetime.timedelta(seconds=start_sec))
        args += ['-ss', ss]
    args += ['-i', input_video, '-s', '224x224']
    if len_sec >= 0:
        to = str(datetime.timedelta(seconds=len_sec))
        args += ['-to', to]

    args += ['-r', str(fps), os.path.join(output_frames_path, '%06d.jpg')]
    return safe_subprocess_call(args)


def clear_artifacts(args):
    files = glob.glob(os.path.join(args.output_path, '*.part'))
    for file in files:
        os.remove(file)

    files = glob.glob(os.path.join(args.output_path, '*.mp4'))
    for file in files:
        os.remove(file)


def read_music_dataset(args):
    dataset = []

    with open(args.dataset_filename, 'r') as f:
        dataset_json = json.load(f)
    videos = dataset_json['videos']

    for label, video_list in videos.items():
        for video in video_list:
            di = DownloadItem(video, label, -1, -1)
            dataset.append(di)

    return dataset


def read_vggsound_dataset(args):
    dataset = []

    with open(args.dataset_filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            di = DownloadItem(row[0], row[2], int(row[1]), 10)
            dataset.append(di)

    return dataset


def read_dataset(args):
    if args.dataset_name == 'MUSIC':
        return read_music_dataset(args)
    if args.dataset_name == 'VGGSound':
        return read_vggsound_dataset(args)
    logger.critical('Unsupported dataset %s', args.dataset_name)
    exit(1)


def try_download(youtube_id, output_path, attempts=3):
    for i in range(attempts):
        try:
            download_video([youtube_id], output_path)
            return True
        except Exception as e:
            if i == attempts - 1:
                logger.warning('Download error %s', e)
                return False
            s = random.randint(1, 10 * (i + 1)) / 10
            time.sleep(s)
    return False


def get_object_name(di: DownloadItem, dataset_name: str):
    if dataset_name == 'MUSIC':
        return f'{di.label}-{di.youtube_id}'
    if dataset_name == 'VGGSound':
        return f'{di.label}-{di.youtube_id}-{di.start_seconds}'
    logger.critical('Unsupported dataset %s', dataset_name)
    exit(1)


def get_audio_frames_path(di, dataset_name, output_path):
    object_name = get_object_name(di, dataset_name)
    img_path = os.path.join(output_path, 'frames')
    audio_path = os.path.join(output_path, 'audio')
    output_audio_filename = os.path.join(audio_path, object_name + '.mp3')
    output_frames_path = os.path.join(img_path, object_name)
    audio_exists = os.path.exists(output_audio_filename)
    frames_exists = os.path.exists(output_frames_path)
    if audio_exists and frames_exists:
        logger.info('%s already downloaded', object_name)
        return None
    if audio_exists:
        os.remove(output_audio_filename)
    if frames_exists:
        shutil.rmtree(output_frames_path)
    return output_frames_path, output_audio_filename


def get_video_filename(di, output_path):
    cache_path = os.path.join(output_path, 'cache', str(os.getpid()))
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    return cache_path, os.path.join(cache_path, di.youtube_id + '.mp4')


def get_fps(dataset_name):
    if dataset_name == 'MUSIC':
        return 8
    if dataset_name == 'VGGSound':
        return 1
    logger.critical('Unsupported dataset %s', dataset_name)
    exit(1)


def download_item(di: DownloadItem, output_path: str, dataset_name: str):
    video_filename = ''
    try:
        audio_frames_path = get_audio_frames_path(di, dataset_name, output_path)
        if not audio_frames_path:
            # already downloaded
            return

        output_frames_path, output_audio_filename = audio_frames_path
        cache_path, video_filename = get_video_filename(di, output_path)

        if os.path.exists(video_filename):
            os.remove(video_filename)

        if not try_download(di.youtube_id, cache_path):
            return

        os.mkdir(output_frames_path)

        if not all([
            extract_audio_ffmpeg(video_filename, output_audio_filename, di.start_seconds, di.len_sec),
            extract_frames_ffmpeg(video_filename, output_frames_path, get_fps(dataset_name), di.start_seconds, di.len_sec)
        ]):
            if os.path.exists(output_audio_filename):
                os.remove(output_audio_filename)
            shutil.rmtree(output_frames_path, ignore_errors=True)
    finally:
        if os.path.exists(video_filename):
            os.remove(video_filename)


def download_dataset(args):
    clear_artifacts(args)
    dataset = read_dataset(args)

    f = partial(download_item, output_path=args.output_path, dataset_name=args.dataset_name)
    with mp.Pool(processes=args.n_jobs) as p:
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for _ in p.imap_unordered(f, dataset):
                pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', help='MUSIC/VGGSound')
    parser.add_argument('dataset_filename', help='Path and filename of vggsound .csv file')
    parser.add_argument('output_path', help='Path to output directory')
    parser.add_argument('--continue_download', help='Keep already downloaded files', default=True, type=bool)
    parser.add_argument('--n_jobs', help='Number of threads', default=1, type=int)
    args = parser.parse_args()

    if not args.continue_download and os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)

    img_path = os.path.join(args.output_path, 'frames')
    audio_path = os.path.join(args.output_path, 'audio')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
        os.mkdir(img_path)
        os.mkdir(audio_path)

    cache_path = os.path.join(args.output_path, 'cache')
    shutil.rmtree(cache_path, ignore_errors=True)
    os.mkdir(cache_path)

    download_dataset(args)
