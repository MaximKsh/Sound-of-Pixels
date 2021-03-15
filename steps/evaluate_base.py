import os

import numpy as np
import torch
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile
from imageio import imwrite

from helpers.utils import makedirs, AverageMeter, warpgrid, istft_reconstruction, magnitude2heatmap, recover_rgb, \
    save_video, combine_video_audio
from helpers.viz import HTMLVisualizer, plot_loss_metrics


def calc_metrics(batch_data, outputs, context):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = context['config']['num_mix']
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if context['config']['log_freq']:
            w = warpgrid(B, context['config']['stft_frame'] // 2 + 1, pred_masks_[0].size(3), warp=False)
            grid_unwarp = torch.from_numpy(w).to(context['device'])
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp, align_corners=True)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if context['config']['binary_mask']:
            pred_masks_linear[n] = (pred_masks_linear[n] > context['config']['mask_thres']).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=context['config']['stft_hop'])

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=context['config']['stft_hop'])

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


def output_visuals(vis_rows, batch_data, outputs, context):
    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    vis = context['vis']
    aud_rate = context['config']['aud_rate']

    # unwarp log scale
    N = context['config']['num_mix']
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    gt_masks_linear = [None for n in range(N)]
    for n in range(N):
        if context['config']['log_freq']:
            w = warpgrid(B, context['config']['stft_frame'] // 2 + 1, gt_masks_[0].size(3), warp=False)
            grid_unwarp = torch.from_numpy(w).to(context['device'])
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp, align_corners=True)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp, align_corners=True)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if context['config']['binary_mask']:
            pred_masks_[n] = (pred_masks_[n] > context['config']['mask_thres']).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > context['config']['mask_thres']).astype(np.float32)

    # loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(vis, prefix))

        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=context['config']['stft_hop'])
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imwrite(os.path.join(vis, filename_mixmag), mix_amp[::-1, :, :])
        imwrite(os.path.join(vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(vis, filename_mixwav), aud_rate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=context['config']['stft_hop'])
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=context['config']['stft_hop'])

            # output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n + 1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n + 1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imwrite(os.path.join(vis, filename_gtmask), gt_mask[::-1, :])
            imwrite(os.path.join(vis, filename_predmask), pred_mask[::-1, :])

            # ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n + 1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n + 1))
            gt_mag = magnitude2heatmap(gt_mag)
            pred_mag = magnitude2heatmap(pred_mag)
            imwrite(os.path.join(vis, filename_gtmag), gt_mag[::-1, :, :])
            imwrite(os.path.join(vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n + 1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n + 1))
            wavfile.write(os.path.join(vis, filename_gtwav), aud_rate, gt_wav)
            wavfile.write(os.path.join(vis, filename_predwav), aud_rate, preds_wav[n])

            # output video
            frames_tensor = [recover_rgb(frames[n][j, :, t]) for t in range(context['config']['num_frames'])]
            frames_tensor = np.asarray(frames_tensor)
            path_video = os.path.join(vis, prefix, 'video{}.mp4'.format(n + 1))
            save_video(path_video, frames_tensor,
                       fps=context['config']['frame_rate'] / context['config']['stride_frames'])

            # combine gt video and audio
            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n + 1))
            combine_video_audio(
                path_video,
                os.path.join(vis, filename_gtwav),
                os.path.join(vis, filename_av))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)


def _evaluate(context: dict):
    epoch = context["epoch"]
    print(f'Evaluating at {context["epoch"]} epochs...')
    makedirs(context['vis'], remove=True)

    net_wrapper = context['net_wrapper']
    net_wrapper.eval()
    loader = context['loader_val']

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # initialize HTML header
    visualizer = HTMLVisualizer(os.path.join(context['vis'], 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, context['config']['num_mix'] + 1):
        header += [f'Video {n:d}', f'Predicted Audio {n:d}', f'GroundTruth Audio {n}', f'Predicted Mask {n}',
                   f'GroundTruth Mask {n}']
    header += ['Loss weighting']
    visualizer.add_header(header)
    vis_rows = []

    for i, batch_data in enumerate(loader):
        err, outputs = net_wrapper.forward(batch_data, context)
        err = err.mean()

        loss_meter.update(err.item())
        print(f'[Eval] iter {i}, loss: {err.item():.4f}')

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, context)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        # output visualization
        if len(vis_rows) < context['config']['num_vis']:
            output_visuals(vis_rows, batch_data, outputs, context)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
          .format(epoch, loss_meter.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average()))

    history = context['history']
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    print('Plotting html for visualization...')
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(context['path'], history)
