import argparse
import math
import os
import shutil
import tempfile
import random
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision
from scipy.io import wavfile
import torchvision.transforms as T
from dataset.music import MUSICMixDataset
from helpers.utils import create_context, read_config, get_ctx, extract_frames_ffmpeg, extract_audio_ffmpeg, \
    istft_reconstruction, save_video, combine_video_audio
from steps.common import build_model, detach_mask, unwarp_log_scale

BASE_FPS = 30


def predict_masks(audio_fname, frame_low_path):
    ctx['load_best_model'] = True
    ctx['net_wrapper'] = build_model(ctx)
    ctx['num_mix'] = 1
    sample = [[audio_fname, frame_low_path, len(os.listdir(frame_low_path))]]

    dataset = MUSICMixDataset(sample, ctx, max_sample=1, split='regions')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    with torch.no_grad():
        data = next(iter(loader))
        output = ctx['net_wrapper'].forward(data, ctx, pixelwise=True)
    return data, output


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def predict_bboxes(img):
    t = get_transform(False)
    input = t(img)
    input = input[None].to(get_ctx(ctx, 'device'))
    with torch.no_grad():
        out = get_ctx(ctx, 'detector')(input)[0]

    boxes = torchvision.ops.nms(out['boxes'], out['scores'], 0.4)
    filtered_boxes = []
    for i in boxes:
        box = out['boxes'][i]
        label = out['labels'][i]
        score = out['scores'][i]
        if label == 1 and score > 0.7:
            filtered_boxes.append(box)

    return filtered_boxes


def get_img_for_detection(path: str):
    files = os.listdir(path)
    filename = sorted([f for f in files if os.path.splitext(f)[1] == '.jpg'])[0]
    return Image.open(os.path.join(path, filename))


def get_mask_boxes(img, grid_width, grid_height):
    width, height = img.size
    b = []
    step_w = width // grid_width
    step_h = height // grid_height
    for x in range(0, width, step_w):
        b.append([])
        for y in range(0, width, step_h):
            b[-1].append([x, y, x + step_w, y + grid_height])
    return b


def area(x_left, y_top, x_right, y_bottom):
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def bb_intersection(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def get_average_mask(box_np, mask_boxes, masks_np, cell_area, mask_width, mask_height, grid_width, grid_height):
    aggregated_mask = np.zeros((mask_width, mask_height), dtype=np.float32)
    d = 0
    for mask_box_x in range(grid_width):
        for mask_box_y in range(grid_height):
            intersection_coef = bb_intersection(mask_boxes[mask_box_x][mask_box_y], box_np) / cell_area
            if intersection_coef != 0:
                aggregated_mask += masks_np[mask_box_x, mask_box_y] * intersection_coef
                # aggregated_mask += masks_np[grid_width - mask_box_x - 1, mask_box_y] * intersection_coef
                d += intersection_coef
    aggregated_mask /= d
    return aggregated_mask


def last_frame_number(path):
    files = os.listdir(path)
    number = sorted([int(os.path.splitext(f)[0]) for f in files])[-1]
    return number


def resize_to_aud_len(wav):
    if get_ctx(ctx, 'aud_len') > wav.shape[0]:
        return np.pad(wav, (0, get_ctx(ctx, 'aud_len') - wav.shape[0]))
    elif get_ctx(ctx, 'aud_len') < wav.shape[0]:
        return wav[:get_ctx(ctx, 'aud_len')]
    return wav


def main():
    ctx['temp_dir'] = tempfile.mkdtemp()
    print(get_ctx(ctx, 'temp_dir'))

    sound_len = int(math.ceil(get_ctx(ctx, 'aud_len') / get_ctx(ctx, 'aud_rate')))
    print(sound_len)

    audio_fname = os.path.join(get_ctx(ctx, 'temp_dir'), 'audio.mp3')
    frame_path = os.path.join(get_ctx(ctx, 'temp_dir'), 'frame')
    os.mkdir(frame_path)
    frame_low_path = os.path.join(get_ctx(ctx, 'temp_dir'), 'frame_low')
    os.mkdir(frame_low_path)

    extract_frames_ffmpeg(get_ctx(ctx, 'video_path'), frame_path, BASE_FPS, get_ctx(ctx, 'time'), sound_len)
    extract_frames_ffmpeg(get_ctx(ctx, 'video_path'), frame_low_path, get_ctx(ctx, 'frame_rate') + 1, get_ctx(ctx, 'time'), sound_len)
    extract_audio_ffmpeg(get_ctx(ctx, 'video_path'), audio_fname, get_ctx(ctx, 'time'), sound_len)

    img = get_img_for_detection(frame_low_path)
    img_width, img_height = img.size
    bboxes = predict_bboxes(img)

    data, out = predict_masks(audio_fname, frame_low_path)
    pred_masks = out['pred_masks']
    _, grid_width, grid_height, mask_width, mask_height = out['pred_masks'].size()

    # unwarp
    pred_masks_linear = torch.zeros((grid_width, 1, grid_height, 512, 256)).to(get_ctx(ctx, 'device'))
    for h in range(grid_width):
        pred_masks_linear_h = unwarp_log_scale(ctx, [pred_masks[:, h, :, :, :]])
        pred_masks_linear[h] = pred_masks_linear_h[0]
    pred_masks_linear = pred_masks_linear.permute(1, 0, 2, 3, 4)
    pred_masks_linear = detach_mask(ctx, [pred_masks_linear], get_ctx(ctx, 'binary_mask'))[0]

    frame_out_path = os.path.join(get_ctx(ctx, 'temp_dir'), 'frame_out')
    shutil.copytree(frame_path, frame_out_path)

    mag_mix = data['mag_mix'].numpy()
    phase_mix = data['phase_mix'].numpy()
    mix_wav = istft_reconstruction(mag_mix[0, 0], phase_mix[0, 0], hop_length=get_ctx(ctx, 'stft_hop'))
    mix_wav = resize_to_aud_len(mix_wav)
    wavfile.write(os.path.join(get_ctx(ctx, 'temp_dir'), 'mix.mp3'), get_ctx(ctx, 'aud_rate'), mix_wav)

    mask_boxes = get_mask_boxes(img, grid_width, grid_height)
    cell_area = area(0, 0, img_width // grid_width, img_height // grid_height)
    for box in bboxes:
        number = last_frame_number(frame_out_path)
        box_np = box.cpu().numpy()
        avg_mask = get_average_mask(box_np, mask_boxes, pred_masks_linear[0], cell_area, 512, 256, grid_width, grid_height)
        if get_ctx(ctx, 'binary_mask'):
            avg_mask = (avg_mask > get_ctx(ctx, 'mask_thres')).astype(np.float32)

        pred_mag = mag_mix[0, 0] * avg_mask
        preds_wav = istft_reconstruction(pred_mag, phase_mix[0, 0], hop_length=get_ctx(ctx, 'stft_hop'))
        preds_wav = resize_to_aud_len(preds_wav)
        wavfile.write(os.path.join(get_ctx(ctx, 'temp_dir'), f'{number:06d}.mp3'), get_ctx(ctx, 'aud_rate'), preds_wav)
        mix_wav = np.concatenate([mix_wav, preds_wav])

        for frame_name in sorted(os.listdir(frame_path)):
            with Image.open(os.path.join(frame_out_path, frame_name)) as im:
                draw = ImageDraw.Draw(im)
                draw.rectangle(box.tolist(), outline='red')

                number += 1
                im.save(os.path.join(frame_out_path, f'{number:06d}.jpg'))

    all_frames = np.array([np.array(Image.open(os.path.join(frame_out_path, frame_name)))
                           for frame_name in sorted(os.listdir(frame_out_path))])
    wavfile.write(os.path.join(get_ctx(ctx, 'temp_dir'), 'full.mp3'), get_ctx(ctx, 'aud_rate'), mix_wav)
    save_video(os.path.join(get_ctx(ctx, 'temp_dir'), 'full.mp4'), all_frames, BASE_FPS)
    combine_video_audio(os.path.join(get_ctx(ctx, 'temp_dir'), 'full.mp4'),
                        os.path.join(get_ctx(ctx, 'temp_dir'), 'full.mp3'),
                        os.path.join(get_ctx(ctx, 'temp_dir'), 'result.mp4'))

    shutil.copy(os.path.join(get_ctx(ctx, 'temp_dir'), 'result.mp4'), get_ctx(ctx, 'output'))

    shutil.rmtree(ctx['temp_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str)
    parser.add_argument('--time', type=int, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='result.mp4')
    args = parser.parse_args()

    config = read_config('configs/default.json')
    if args.config != '':
        config_override = read_config(args.config)
        config = {**config, **config_override}

    ctx = create_context(config)
    ctx['video_path'] = args.video
    ctx['time'] = args.time
    ctx['output'] = args.output

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    ctx['detector'] = model.to(get_ctx(ctx, 'device')).eval()

    random.seed(get_ctx(ctx, 'seed'))
    np.random.seed(get_ctx(ctx, 'seed'))
    torch.manual_seed(get_ctx(ctx, 'seed'))

    main()
