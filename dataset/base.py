import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torchaudio
import librosa
from PIL import Image

from helpers.utils import get_ctx
from . import video_transforms as vtransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, ctx, max_sample=-1, split='train'):
        # params
        self.num_frames = get_ctx(ctx, 'num_frames')
        self.stride_frames = get_ctx(ctx, 'stride_frames')
        self.frame_rate = get_ctx(ctx, 'frame_rate')
        self.img_size = get_ctx(ctx, 'img_size')
        self.aud_rate = get_ctx(ctx, 'aud_rate')
        self.aud_len = get_ctx(ctx, 'aud_len')
        self.aud_sec = 1. * self.aud_len / self.aud_rate
        self.binary_mask = get_ctx(ctx, 'binary_mask')

        # STFT params
        self.log_freq = get_ctx(ctx, 'log_freq')
        self.stft_frame = get_ctx(ctx, 'stft_frame')
        self.stft_hop = get_ctx(ctx, 'stft_hop')
        self.HS = get_ctx(ctx, 'stft_frame') // 2 + 1
        self.WS = (self.aud_len + 1) // self.stft_hop

        self.split = split

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            # self.list_sample = [x.rstrip() for x in open(list_sample, 'r')]
            self.list_sample = []
            with open(list_sample, 'r') as f:
                for row in csv.reader(f, delimiter=','):
                    if len(row) < 2:
                        continue
                    self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise RuntimeError('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= get_ctx(ctx, 'dup_trainset')
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, index):
        raise NotImplementedError()

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.img_size * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.img_size))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.img_size, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.img_size))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.img_size * 1.2)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        if path.endswith('.mp3'):
            audio_raw, rate = torchaudio.load(path)
            audio_raw = audio_raw.numpy().astype(np.float32)
     
            # range to [-1, 1]
            #audio_raw *= (2.0**-31)

            # convert to mono
            if audio_raw.shape[1] == 2:
                audio_raw = (audio_raw[0, :] + audio_raw[1, :]) / 2
            else:
                audio_raw = audio_raw[0, :]
        else:
            audio_raw, rate = librosa.load(path, sr=None, mono=True)

        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.aud_len, dtype=np.float32)
        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.aud_sec:
            n = int(rate * self.aud_sec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.aud_rate:
            print('resmaple {}->{}'.format(rate, self.aud_rate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.aud_rate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.aud_rate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.aud_rate)
        start = max(0, center - self.aud_len // 2)
        end = min(len_raw, center + self.aud_len // 2)

        audio[self.aud_len // 2 - (center - start): self.aud_len // 2 + (end - center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.

        return audio

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        # to tensor
        # audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.img_size, self.img_size)
            audios[n] = torch.zeros(self.aud_len)
            mags[n] = torch.zeros(1, self.HS, self.WS)

        return amp_mix, mags, frames, audios, phase_mix
