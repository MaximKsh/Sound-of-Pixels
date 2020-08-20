# System libs
import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
from imageio import imwrite
from mir_eval.separation import bss_eval_sources
from torchsummary import summary

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    combine_video_audio, save_video, makedirs

class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_synthesizer = nets

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        # mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        mag_mix = mag_mix.to(args.device)
        for i in range(len(frames)):
            frames[i] = frames[i].to(args.device)
        
        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_sound -> BxCxHxW
        feat_sound = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, args.sound_activation)
        
        print(feat_sound.shape)
        print()
        
        # 2. forward net_frame -> Bx1xC
        print(frames[0].shape)
        feat_frames = self.net_frame.forward_multiframe(frames[0], pool=False)
        print(feat_frames.shape)
        feat_frames = torch.max(feat_frames, dim=2)[0]
        print(feat_frames.shape)
        feat_frames = activate(feat_frames, args.img_activation)
        print(feat_frames.shape)
            
        # 3. sound synthesizer
        pred_masks = self.net_synthesizer.forward_pixelwise(feat_frames, feat_sound)
        pred_masks = activate(pred_masks, args.output_activation)
        print(pred_masks.shape)

        return {'pred_masks': pred_masks, 'processed_mag_mix': mag_mix}

def output_predictions(data, outputs, args):
    mag_mix = data['mag_mix']
    phase_mix = data['phase_mix']
    frames = data['frames']
    infos = data['infos']

    #2
    #torch.Size([32, 1, 256, 256])
    #torch.Size([32, 1, 512, 256])
    #torch.Size([32, 512, 256, 2]) torch.Size([32, 1, 256, 256])
    #torch.Size([32, 512, 256, 2]) torch.Size([32, 1, 256, 256])
    
    
    # torch.Size([1, 14, 14, 256, 256])
    # torch.Size([1, 1, 512, 256])

    
    pred_masks_ = outputs['pred_masks']
    mag_mix_ = outputs['processed_mag_mix']
    #pred_masks_ = pred_masks_[0]
    
    print('pred_masks', pred_masks_.shape)
    print('mag_mix', mag_mix.shape)
    
    print(infos)
    N = args.num_mix
    B = mag_mix.size(0)
    grid_unwarp = torch.from_numpy(warpgrid(B, args.stft_frame//2+1, pred_masks_.size(4), warp=False)).to(args.device)
    pred_masks_linear = [[None for _ in range(14)] for _ in range(14)]
    
    print(grid_unwarp.shape)
    print(pred_masks_[0, 0, 0][None, None].shape)
    
    if args.log_freq:
        for x in range(14):
            for y in range(14):
                pred_masks_linear[x][y] = F.grid_sample(pred_masks_[0, x, y][None, None], grid_unwarp, align_corners=True)
                pred_masks_linear[x][y] = pred_masks_linear[x][y].detach().cpu().numpy()
                
                if args.binary_mask:
                    pred_masks_linear[x][y] = (pred_masks_linear[x][y] > args.mask_thres).astype(np.float32)
    else:
        for x in range(14):
            for y in range(14):
                pred_masks_linear[x][y] = pred_masks_[0, x, y][None, None]
                pred_masks_linear[x][y] = pred_masks_linear[x][y].detach().cpu().numpy()
                
                if args.binary_mask:
                    pred_masks_linear[x][y] = (pred_masks_linear[x][y] > args.mask_thres).astype(np.float32)
        

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    
    pred_masks_ = pred_masks_.detach().cpu().numpy()

    # threshold if binary mask
    if args.binary_mask:
        pred_masks_ = (pred_masks_ > args.mask_thres).astype(np.float32)
        #pred_masks_linear = (pred_masks_linear > args.mask_thres).astype(np.float32)

    j = 0
    prefix = '-'.join(infos[0][0][j].split('/')[-2:]).split('.')[0]
    makedirs(os.path.join(args.vis, prefix))
    
    cellular_pred_mask = np.zeros((14 * 256, 14 * 256))
    
    for x in range(14):
        for y in range(14):
            name = f'{x}x{y}'
            
            # predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[x][y][j, 0]
            preds_wav = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

            # output masks
            filename_predmask = os.path.join(prefix, f'{name}-predmask.jpg')
            pred_mask = (np.clip(pred_masks_[j, x, y], 0, 1) * 255).astype(np.uint8)
            imwrite(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])
            cellular_pred_mask[x * 256:(x+1) * 256, y * 256:(y+1) * 256] = pred_mask[::-1, :]

            # ouput spectrogram (log of magnitude, show colormap)
            filename_predmag = os.path.join(prefix, f'{name}-predamp.jpg')
            pred_mag = magnitude2heatmap(pred_mag)
            imwrite(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            # output audio
            filename_predwav = os.path.join(prefix, f'{name}-pred.wav')
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav)
            
    imwrite(os.path.join(args.vis, os.path.join(prefix, f'cellular-predmask.jpg')), cellular_pred_mask)
    
    
def main(args):
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        fc_dim=args.num_channels,
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_dim=args.num_channels,
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_synthesizer = builder.build_synthesizer(
        arch=args.arch_synthesizer,
        fc_dim=args.num_channels,
        weights=args.weights_synthesizer)
    nets = (net_sound, net_frame, net_synthesizer)
    
    dataset = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')
    loader_val = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    
    netWrapper = NetWrapper(nets)
    netWrapper.eval()
    netWrapper.to(args.device)
    
    #it = iter(loader_val)
    #data = next(it)
    
    
    for data in loader_val:
        output = netWrapper.forward(data, args)
        output_predictions(data, output, args)

    
    
if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")
    print(args.device)
    
    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'test_visualization/')
    args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
    args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
    args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_best.pth')
  
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)
