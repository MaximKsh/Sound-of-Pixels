import os
import random

from helpers.utils import get_ctx
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, ctx, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, ctx, **kwargs)
        self.fps = get_ctx(ctx, 'frame_rate')
        self.num_mix = get_ctx(ctx, 'num_mix')

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for _ in range(N)]
        audios = [None for _ in range(N)]
        infos = [[] for _ in range(N)]
        path_frames = [[] for _ in range(N)]
        path_audios = ['' for _ in range(N)]
        center_frames = [0 for _ in range(N)]

        # the first video
        infos[0] = self.list_sample[index]

        # sample other videos
        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]

        # select frames
        idx_margin = max(int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)
        
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN
            
            if self.split == 'train' and int(count_framesN) > 20: # vggsound hack
                # random, not to sample start and end n-frames
                center_frameN = random.randint(
                    idx_margin+1, int(count_framesN)-idx_margin)
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN


        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                try:
                    frames[n] = self._load_frames(path_frames[n])
                except Exception as e:
                    label = 'load frame'
                    raise e
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                
                try:
                    audios[n] = self._load_audio(path_audios[n], center_timeN)
                except Exception as e:
                    label = 'load audio'
                    raise e
            try:
                mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
            except Exception as e:
                label = 'mix n and stft'
                raise e
            
        except Exception as e:
            print('Failed {}: {}'.format(label, e))
            # create dummy data
            mag_mix, mags, frames, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
