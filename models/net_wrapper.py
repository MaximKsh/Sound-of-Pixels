import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from helpers.utils import warpgrid, get_ctx
from .synthesizer_net import InnerProd, Bias, Transformer
from .audio_net import Unet
from .vision_net import ResnetFC, ResnetDilated
from .criterion import BCELoss, L1Loss, L2Loss


def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return torch.tanh(x)
    elif activation == 'abstanh':
        return torch.abs(torch.tanh(x))
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class NetWrapper(nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_synthesizer = nets
        self.crit = crit

    def forward(self, batch_data, ctx, pixelwise=False):
        if pixelwise:
            return self._forward_pixelwise(batch_data, ctx)
        return self._forward(batch_data, ctx)

    def _forward(self, batch_data, ctx):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10

        N = get_ctx(ctx, 'num_mix')
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if get_ctx(ctx, 'log_freq'):
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(get_ctx(ctx, 'device'))
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=True)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if get_ctx(ctx, 'weighted_loss'):
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if get_ctx(ctx, 'binary_mask'):
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_sound -> BxCxHxW
        feat_sound = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, get_ctx(ctx, 'sound_activation'))

        # 2. forward net_frame -> Bx1xC
        feat_frames = [None for n in range(N)]
        for n in range(N):
            feat_frames[n] = self.net_frame.forward_multiframe(frames[n])
            feat_frames[n] = activate(feat_frames[n], get_ctx(ctx, 'img_activation'))

        # 3. sound synthesizer
        pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_synthesizer(feat_frames[n], feat_sound)
            pred_masks[n] = activate(pred_masks[n], get_ctx(ctx, 'output_activation'))

        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)

        return err, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'weight': weight}

    def _forward_pixelwise(self, batch_data, ctx):
        mag_mix = batch_data['mag_mix']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10

        bs = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if get_ctx(ctx, 'log_freq'):
            grid_warp = torch.from_numpy(warpgrid(bs, 256, T, warp=True)).to(get_ctx(ctx, 'device'))
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=True)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_sound -> BxCxHxW
        feat_sound = self.net_sound(log_mag_mix)
        feat_sound = activate(feat_sound, get_ctx(ctx, 'sound_activation'))

        # 2. forward net_frame -> Bx1xC
        frames = frames[0]  # num_mix == 1
        feat_frames = self.net_frame.forward_multiframe(frames, pool=False)

        (B, C, T, H, W) = feat_frames.size()
        feat_frames = feat_frames.permute(0, 1, 3, 4, 2)
        feat_frames = feat_frames.reshape(B * C, H * W, T)
        feat_frames = F.adaptive_avg_pool1d(feat_frames, 1)
        feat_frames = feat_frames.view(B, C, H, W)

        feat_frames = activate(feat_frames, get_ctx(ctx, 'img_activation'))

        channels = feat_frames.detach().cpu().numpy()

        # 3. sound synthesizer
        pred_masks = self.net_synthesizer.forward_pixelwise(feat_frames, feat_sound)
        pred_masks = activate(pred_masks, get_ctx(ctx, 'output_activation'))

        return {'pred_masks': pred_masks, 'processed_mag_mix': mag_mix, 'feat_frames_channels': channels}


class ModelBuilder:
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_sound(self, arch='unet5', fc_dim=64, weights='', device='cpu'):
        # 2D models
        if arch == 'unet5':
            net_sound = Unet(fc_dim=fc_dim, num_downs=5)
        elif arch == 'unet6':
            net_sound = Unet(fc_dim=fc_dim, num_downs=6)
        elif arch == 'unet7':
            net_sound = Unet(fc_dim=fc_dim, num_downs=7)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights, map_location=device))

        return net_sound

    def build_frame(self, arch='resnet18', fc_dim=64, pool_type='avgpool', weights='', device='cpu'):
        pretrained = True
        if arch == 'resnet18fc':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetFC(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResnetDilated(
                original_resnet, fc_dim=fc_dim, pool_type=pool_type)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights, map_location=device))
        return net

    def build_synthesizer(self, arch, fc_dim=64, weights='', device='cpu'):
        if arch == 'linear':
            net = InnerProd(fc_dim=fc_dim)
        elif arch == 'bias':
            net = Bias()
        elif arch == 'transformer':
            net = Transformer(fc_dim)
        else:
            raise Exception('Architecture undefined!')

        net.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_synthesizer')
            net.load_state_dict(torch.load(weights, map_location=device))
        return net

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Architecture undefined!')
        return net
