import copy
import json
import os
import shutil

import numpy as np
import librosa
import cv2

import subprocess as sp
from threading import Timer

import torch


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def makedirs(path, remove=False, verbose=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            if verbose:
                print(f'Remove: {path}')
        else:
            if verbose:
                print(f'Already exists: {path}')
            return
    if verbose:
        print(f'Create: {path}')
    os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()


def recover_rgb(img):
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


class VideoWriter:
    """ Combine numpy frames into video using ffmpeg

    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame

    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe

    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split('.')[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",                                     # overwrite existing file
            "-f", "rawvideo",                         # file format
            "-s", "{}x{}".format(shape[1], shape[0]), # size of one frame
            "-pix_fmt", "rgb24",                      # 3 channels
            "-r", str(self.fps),                      # frames per second
            "-i", "-",                                # input comes from a pipe
            "-an",                                    # not to expect any audio
            "-vcodec", self.vcodec,                   # video codec
            "-pix_fmt", "yuv420p",                  # output video in yuv420p
            self.file]

        self.pipe = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10**9)

    def release(self):
        self.pipe.stdin.close()

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            self.pipe.stdin.write(frame.tostring())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)


def kill_proc(proc):
    proc.kill()
    print('Process running overtime! Killed.')


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = ["ffmpeg", "-y",
               "-loglevel", "quiet",
               "-i", src_video,
               "-i", src_audio,
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               dst_video]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.)

        if verbose:
            print('Processed:{}'.format(dst_video))
    except Exception as e:
        print('Error:[{}] {}'.format(dst_video, e))


# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    assert tensor.ndim == 4, 'video should be in 4D numpy array'
    L, H, W, C = tensor.shape
    writer = VideoWriter(
        path,
        fps=fps,
        shape=[H, W])
    
    for t in range(L):
        writer.add_frame(tensor[t])
        
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)


def format_id(config: dict) -> str:
    id_ = config['id']

    if config['finetune']:
        id_ += f'-finetune{hash(config["finetune"])}'

    id_ += f'-{config["num_mix"]}mix'

    if config['log_freq']:
        id_ += '-LogFreq'

    id_ += f'-{config["arch_frame"]}{config["img_activation"]}-' \
           f'{config["arch_sound"]}{config["sound_activation"]}-' \
           f'{config["arch_synthesizer"]}{config["output_activation"]}'
    id_ += f'-frames{config["num_frames"]}stride{config["stride_frames"]}'
    id_ += f'-{config["img_pool"]}'

    if config['binary_mask']:
        assert config['loss'] == 'bce', 'Binary Mask should go with BCE loss'
        id_ += '-binary'
    else:
        id_ += '-ratio'

    if config['weighted_loss']:
        id_ += '-weightedLoss'
    id_ += f'-channels{config["num_channels"]}'
    id_ += f'-epoch{config["num_epoch"]}'

    id_ += '-step' + '_'.join([str(x) for x in config['lr_steps']])

    return id_


def read_config(path: str) -> dict:
    with open(path, "r") as f:
        config_json = json.load(f)
    return config_json


def create_context(config: dict) -> dict:
    context = {
        'config': config,
        'batch_size': min(1, len(config['gpu'])) * config['batch_size_per_gpu'],
        'id': format_id(config)
    }

    if len(config['gpu']) > 0 and torch.cuda.is_available():
        context['device'] = torch.device('cuda')
    else:
        context['device'] = torch.device('cpu')

    context['path'] = os.path.join(config['ckpt'], context['id'])
    context['vis_val'] = os.path.join(context['path'], config['val_vis_dir'])
    context['vis_test'] = os.path.join(context['path'], config['test_vis_dir'])
    context['vis_regions'] = os.path.join(context['path'], config['regions_vis_dir'])
    context['load_best_model'] = False

    context['weights_sound_latest'] = os.path.join(context['path'], 'sound_latest.pth')
    context['weights_frame_latest'] = os.path.join(context['path'], 'frame_latest.pth')
    context['weights_synthesizer_latest'] = os.path.join(context['path'], 'synthesizer_latest.pth')

    if config['finetune']:
        context['weights_sound_finetune'] = os.path.join(config['finetune'], 'sound_latest.pth')
        context['weights_frame_finetune'] = os.path.join(config['path'], 'frame_latest.pth')
        context['weights_synthesizer_finetune'] = os.path.join(config['path'], 'synthesizer_latest.pth')

    context['weights_sound_best'] = os.path.join(context['path'], 'sound_best.pth')
    context['weights_frame_best'] = os.path.join(context['path'], 'frame_best.pth')
    context['weights_synthesizer_best'] = os.path.join(context['path'], 'synthesizer_best.pth')

    context['lr_sound'] = config['lr_sound']
    context['lr_frame'] = config['lr_frame']
    context['lr_synthesizer'] = config['lr_synthesizer']

    context['best_err'] = float('inf')
    return context


def get_ctx(context: dict, key: str):
    try:
        return context[key]
    except KeyError:
        return context['config'][key]
