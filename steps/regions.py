import os

import torch
from imageio import imwrite
from scipy.io import wavfile
from tqdm import tqdm

import numpy as np

from dataset.music import MUSICMixDataset
from helpers.utils import get_ctx, recover_rgb, makedirs, istft_reconstruction, magnitude2heatmap
from steps.common import build_model, unwarp_log_scale, detach_mask
from steps.regions_report_template import sbr_html


def output_predictions(ctx, data, outputs):
    mag_mix = data['mag_mix']
    phase_mix = data['phase_mix']
    frames = data['frames']
    infos = data['infos']
    pred_masks_ = outputs['pred_masks']

    bs, im_h, im_w, _, _ = pred_masks_.shape

    # unwarp
    pred_masks_linear = torch.zeros((im_h, bs, im_w, 512, 256)).to(get_ctx(ctx, 'device'))
    for h in range(im_h):
        pred_masks_linear_h = unwarp_log_scale(ctx, [pred_masks_[:, h, :, :, :]])
        pred_masks_linear[h] = pred_masks_linear_h[0]
    pred_masks_linear = pred_masks_linear.permute(1, 0, 2, 3, 4)

    # to cpu
    pred_masks_linear = detach_mask(ctx, [pred_masks_linear], get_ctx(ctx, 'binary_mask'))[0]
    pred_masks_ = detach_mask(ctx, [pred_masks_], get_ctx(ctx, 'binary_mask'))[0]
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    frames = frames[0]

    for i in range(bs):
        frames_tensor = np.asarray([recover_rgb(frames[i, :, t].cpu()) for t in range(get_ctx(ctx, 'num_frames'))])

        pth, id_ = os.path.split(infos[0][1][i])
        _, group = os.path.split(pth)
        prefix = group + '-' + id_
        folder = os.path.join(get_ctx(ctx, 'vis_regions'), prefix)
        sbr_folder = os.path.join(folder, 'sbr')
        grid_folder = os.path.join(sbr_folder, 'grid')

        makedirs(folder)
        makedirs(sbr_folder)
        makedirs(grid_folder)

        grid_pred_mask = np.zeros((14 * 256, 14 * 256))

        for j in range(get_ctx(ctx, 'num_frames')):
            imwrite(os.path.join(folder, f'frame{j}.jpg'), frames_tensor[j])

        mix_wav = istft_reconstruction(mag_mix[i, 0], phase_mix[i, 0], hop_length=get_ctx(ctx, 'stft_hop'))
        wavfile.write(os.path.join(folder, 'mix.wav'), get_ctx(ctx, 'aud_rate'), mix_wav)

        # SBR
        for h in range(im_h):
            for w in range(im_w):
                name = f'{h}x{w}'

                # output audio
                pred_mag = mag_mix[i, 0] * pred_masks_linear[i, h, w]
                preds_wav = istft_reconstruction(pred_mag, phase_mix[i, 0], hop_length=get_ctx(ctx, 'stft_hop'))
                wavfile.write(os.path.join(grid_folder, f'{name}-pred.wav'), get_ctx(ctx, 'aud_rate'), preds_wav)

                # output masks
                pred_mask = (np.clip(pred_masks_[i, h, w], 0, 1) * 255).astype(np.uint8)
                imwrite(os.path.join(grid_folder, f'{name}-predmask.jpg'), pred_mask[::-1, :])
                grid_pred_mask[h * 256:(h + 1) * 256, w * 256:(w + 1) * 256] = pred_mask[::-1, :]

                # ouput spectrogram (log of magnitude, show colormap)
                pred_mag = magnitude2heatmap(pred_mag)
                imwrite(os.path.join(grid_folder, f'{name}-predamp.jpg'), pred_mag[::-1, :, :])

        imwrite(os.path.join(sbr_folder, f'masks-grid.jpg'), grid_pred_mask)

        grid_frame = frames_tensor[0]
        grid_frame[:, np.arange(16, 224, 16)] = 255
        grid_frame[np.arange(16, 224, 16), :] = 255
        imwrite(os.path.join(sbr_folder, f'frame.jpg'), grid_frame)

        with open(os.path.join(sbr_folder, 'sbr.html'), 'w') as text_file:
            text_file.write(sbr_html)


def regions(ctx: dict):
    ctx['load_best_model'] = True
    ctx['net_wrapper'] = build_model(ctx)
    ctx['num_mix'] = 1

    dataset = MUSICMixDataset(get_ctx(ctx, 'list_regions'), ctx, max_sample=get_ctx(ctx, 'num_regions'),
                              split='regions')

    loader = torch.utils.data.DataLoader(dataset, batch_size=get_ctx(ctx, 'batch_size'), shuffle=True,
                                         num_workers=1, drop_last=False)

    makedirs(get_ctx(ctx, 'vis_regions'), remove=True)
    cnt = 0
    with torch.no_grad():
        for data in tqdm(loader):
            output = ctx['net_wrapper'].forward(data, ctx, pixelwise=True)
            output_predictions(ctx, data, output)
            cnt += len(data['audios'][0])
