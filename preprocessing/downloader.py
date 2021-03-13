import youtube_dl
import os
import shutil
import json
import argparse


def download_video(videos, output_path):
    ydl_opts = {
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'keepvideo': True,
        'ignoreerrors': True,
        'format': 'mp4',
        # 'postprocessors': [{
        #     'key': 'FFmpegExtractAudio',
        #     'preferredcodec': 'mp3',
        # }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(videos)


def download_dataset(args):
    for group in groups:
        group_filename = os.path.join(args.metadata_path, f'{group}.json')
        with open(group_filename, 'r') as file:
            metainfo = json.loads(file.read())

        group_output_path = os.path.join(args.output_path, group)
        os.mkdir(group_output_path)
        for category in metainfo['videos']:
            output_dir = os.path.join(group_output_path, category) 
            os.mkdir(output_dir)
            
            files = metainfo['videos'][category]
            download_video(files, output_dir)


def get_groups(metadata_path):
    groups = []
    for filename in os.listdir(metadata_path):
        if os.path.isfile(os.path.join(metadata_path, filename)):
            filename_without_ext, ext = os.path.splitext(filename)
            if ext == '.json':
                groups.append(filename_without_ext)
    return groups


def clear_output(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path", help='Path to MUSIC metadata json files')
    parser.add_argument('output_path', help='Path to output directory')
    args = parser.parse_args()

    groups = get_groups(args.metadata_path)
    clear_output(args.output_path)
    print(groups)

    download_dataset(args)
