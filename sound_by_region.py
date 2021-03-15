# System libs
import os
import random

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
from imageio import imwrite
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder, activate
from helpers.utils import recover_rgb, magnitude2heatmap,\
    istft_reconstruction, warpgrid, \
    makedirs

sbr_html = """
<html>
  <head>
    <title></title>
    <meta content="">
    <style></style>
  </head>
  <body>
  
    <table style="width:50%">
        <tr>
            <th><img src="masks-grid.jpg" height=448 width=448/></th>
            <th><img class="frame" src="frame.jpg"  height=448 width=448/></th>
        </tr>
    </table>
    
    <script>
    function offset(el) {
	    var rect = el.getBoundingClientRect(),
	    scrollLeft = window.pageXOffset || document.documentElement.scrollLeft,
	    scrollTop = window.pageYOffset || document.documentElement.scrollTop;
	    return { top: rect.top + scrollTop, left: rect.left + scrollLeft }
	}
	
    function printMousePos(event) {
        var frame = document.getElementsByClassName("frame")[0];
        var pos = offset(frame)
        var x = Math.floor((event.clientX - pos["left"]) / 32);
        var y = Math.floor((event.clientY - pos["top"]) / 32);
    
        console.log("clientX: " + (event.clientX - pos["left"]) + " - clientY: " + (event.clientY - pos["top"]));
        
        sound_filename = "grid/" + y + "x" + x + "-pred.wav";
        console.log(sound_filename);
        var audio = new Audio(sound_filename);
        audio.play();
    }

    var frame = document.getElementsByClassName("frame");
    frame[0].addEventListener("click", printMousePos);
    </script>
    
  </body>
</html>
"""
    
    
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
        
        #print(feat_sound.shape)
        #print()
        
        # 2. forward net_frame -> Bx1xC
        #print(frames[0].shape)
        feat_frames = self.net_frame.forward_multiframe(frames[0], pool=False)
        
        #feat_frames = torch.max(feat_frames, dim=2)[0]
        #print(feat_frames.shape)
        (B, C, T, H, W) = feat_frames.size()
        feat_frames = feat_frames.permute(0, 1, 3, 4, 2)
        feat_frames = feat_frames.view(B*C, H*W, T)
        feat_frames = F.adaptive_avg_pool1d(feat_frames, 1)
        feat_frames = feat_frames.view(B, C, H, W)
        
        #print(feat_frames.shape)
        feat_frames = activate(feat_frames, args.img_activation)
        #print(feat_frames.shape)
        # feat_frames[:, 13, :, :] = 0
        
        
#         feat_frames1 = feat_frames[0, 10, :, :]
#         feat_frames = feat_frames * 1e-10
#         feat_frames[0, 10, :, :] = feat_frames1
            
        channels = feat_frames.detach().cpu().numpy()
        #print(channels)
        #print(channels.shape)
        #print(self.net_synthesizer.scale)
        #print(feat_sound.mean(axis=3).mean(axis=2))
        
        # 3. sound synthesizer
        pred_masks = self.net_synthesizer.forward_pixelwise(feat_frames, feat_sound)
        pred_masks = activate(pred_masks, args.output_activation)
        # print(pred_masks.shape)

        return {'pred_masks': pred_masks, 'processed_mag_mix': mag_mix, 'feat_frames_channels': channels}

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
    channels_ = outputs['feat_frames_channels']
    #pred_masks_ = pred_masks_[0]
    
    #print('pred_masks', pred_masks_.shape)
    ##print('mag_mix', mag_mix.shape)
    
    #print(infos)
    N = args.num_mix
    B = mag_mix.size(0)
    grid_unwarp = torch.from_numpy(warpgrid(B, args.stft_frame//2+1, pred_masks_.size(4), warp=False)).to(args.device)
    pred_masks_linear = [[None for _ in range(14)] for _ in range(14)]
    
    #print(grid_unwarp.shape)
    #print(pred_masks_[0, 0, 0][None, None].shape)
    
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

    j = 0
    frames_tensor = [recover_rgb(frames[0][j, :, t].cpu()) for t in range(args.num_frames)]
    frames_tensor = np.asarray(frames_tensor)
    
    prefix = '-'.join(infos[0][0][j].split('/')[-2:]).split('.')[0]
    folder = os.path.join(args.vis, prefix)
    sbr_folder = os.path.join(folder, 'sbr')
    rbs_folder = os.path.join(folder, 'rbs')
    grid_folder = os.path.join(sbr_folder, 'grid')
    makedirs(folder)
    makedirs(sbr_folder)
    makedirs(rbs_folder)
    makedirs(grid_folder)
    
    grid_pred_mask = np.zeros((14 * 256, 14 * 256))
    
    for i in range(args.num_frames):
        imwrite(os.path.join(folder, f'frame{i}.jpg'), frames_tensor[i])
    mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)
    wavfile.write(os.path.join(folder, 'mix.wav'), args.aud_rate, mix_wav)
    
    # SBR
    for x in range(14):
        for y in range(14):
            name = f'{x}x{y}'
            
            # output audio
            pred_mag = mag_mix[j, 0] * pred_masks_linear[x][y][j, 0]
            preds_wav = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)
            wavfile.write(os.path.join(grid_folder, f'{name}-pred.wav'), args.aud_rate, preds_wav)
            
            # output masks
            pred_mask = (np.clip(pred_masks_[j, x, y], 0, 1) * 255).astype(np.uint8)
            imwrite(os.path.join(grid_folder, f'{name}-predmask.jpg'), pred_mask[::-1, :])
            grid_pred_mask[x * 256:(x+1) * 256, y * 256:(y+1) * 256] = pred_mask[::-1, :]

            # ouput spectrogram (log of magnitude, show colormap)
            pred_mag = magnitude2heatmap(pred_mag)
            imwrite(os.path.join(grid_folder, f'{name}-predamp.jpg'), pred_mag[::-1, :, :])

    imwrite(os.path.join(sbr_folder, f'masks-grid.jpg'), grid_pred_mask)
    
    grid_frame = frames_tensor[0]
    grid_frame[:,np.arange(16, 224, 16)] = 255
    grid_frame[np.arange(16, 224, 16), :] = 255
    imwrite(os.path.join(sbr_folder, f'frame.jpg'), grid_frame)
        
    with open(os.path.join(sbr_folder, 'sbr.html'), 'w') as text_file:
        text_file.write(sbr_html)
    
    # RBS
    
    channels_ = ((np.clip(channels_, -1, 1) + 1) / 2 * 255).astype(np.uint8)
    # channels_[j][channels_[j] < .5] = 0
    all_channels = np.max(channels_[j], axis=0)
    
    plt.figure()
    plt.subplot(5, 8, 1)
    plt.axis('off')
    plt.title('max')
    plt.imshow(all_channels, cmap='RdYlGn')
    
    for i in range(32):
        plt.subplot(5, 8, 9 + i)
        plt.axis('off')
        plt.title(str(i))
        plt.imshow(channels_[j, i], cmap='RdYlGn')
    
    plt.savefig(os.path.join(rbs_folder, f'channels.jpg'))
    
    plt.figure(figsize=(40, 40))
    plt.subplot(5, 8, 1)
    plt.axis('off')
    plt.title('all')
    plt.imshow(frames_tensor[0])
    plt.imshow(zoom(all_channels, 224/14), alpha=0.7, cmap='RdYlGn')
    
    for i in range(32):
        plt.subplot(5, 8, 9 + i)
        plt.axis('off')
        plt.title(str(i))
        plt.imshow(frames_tensor[0])
        plt.imshow(zoom(channels_[j, i], 224/14), alpha=0.7, cmap='RdYlGn')
    
    plt.savefig(os.path.join(rbs_folder, f'frames_channels.jpg'))
    
    #for c in range(channels_.shape[1]):
    #    imwrite(os.path.join(channels_folder, f'channel{c}.jpg'), channels_[j, c])
        
    #imwrite(os.path.join(rbs_folder, f'all_channels.jpg'), )
     
    
    
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
        dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)
    
    netWrapper = NetWrapper(nets)
    netWrapper.eval()
    netWrapper.to(args.device)
    
    cnt = 0
    for data in loader_val:
        output = netWrapper.forward(data, args)
        output_predictions(data, output, args)
        if cnt == 50:
            break
        cnt += 1


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")
    print(args.device)
    
    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, args.vis_dir)
    args.weights_sound = os.path.join(args.ckpt, 'sound_latest.pth')
    args.weights_frame = os.path.join(args.ckpt, 'frame_latest.pth')
    args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_latest.pth')
  
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)
