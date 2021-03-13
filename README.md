# Sound-of-Pixels


## Requirements and installation
Python: 3.8.7, CUDA: 11.0

Install requirements CPU only:
```
python3 -m pip install -r requirements.txt
```

Install requirements GPU:
```
python3 -m pip install -r requirements-gpu.txt -f https://download.pytorch.org/whl/torch_stable.html
```


## Data collection and preprocessing

Data downloading and preprocessing will be described later...

Links with preprocessed data:
* MUSIC:
* VGGSound: 

## Indexing dataset

Training process requires input data described in csv format:
```
 ./data/audio/acoustic_guitar/M3dekVSwNjY.mp3,./data/frames/acoustic_guitar/M3dekVSwNjY.mp4,1580
 ./data/audio/trumpet/STKXyBGSGyE.mp3,./data/frames/trumpet/STKXyBGSGyE.mp4,493
```
You can generate that files using create_index_files.py (run it from repository root):
* MUSIC:
```
python3 preprocessing/create_index_files.py <path-to-dataset>/sop-data/solo21+duet+silence music 
```
* VGGSound 
```
python3 preprocessing/create_index_files.py <path-to-dataset>/vggsound vggsound --min_frames=0 
```


## Training
1. Prepare video dataset.

    a. Download MUSIC dataset from: https://github.com/roudimit/MUSIC_dataset
    
    b. Download videos.

2. Preprocess videos. You can do it in your own way as long as the index files are similar.

    a. Extract frames at 8fps and waveforms at 11025Hz from videos. We have following directory structure:
    ```
    data
    ├── audio
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp3
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp3
    │   |   ├── ...
    │   ├── ...
    |
    └── frames
    |   ├── acoustic_guitar
    │   |   ├── M3dekVSwNjY.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── trumpet
    │   |   ├── STKXyBGSGyE.mp4
    │   |   |   ├── 000001.jpg
    │   |   |   ├── ...
    │   |   ├── ...
    │   ├── ...
    ```

    b. Make training/validation index files by running:
    ```
    python scripts/create_index_files.py
    ```
    It will create index files ```train.csv```/```val.csv``` with the following format:
    ```
    ./data/audio/acoustic_guitar/M3dekVSwNjY.mp3,./data/frames/acoustic_guitar/M3dekVSwNjY.mp4,1580
    ./data/audio/trumpet/STKXyBGSGyE.mp3,./data/frames/trumpet/STKXyBGSGyE.mp4,493
    ```
    For each row, it stores the information: ```AUDIO_PATH,FRAMES_PATH,NUMBER_FRAMES```

3. Train the default model.
```bash
./scripts/train_MUSIC.sh
```

5. During training, visualizations are saved in HTML format under ```ckpt/MODEL_ID/visualization/```.

## Evaluation
0. (Optional) Download our trained model weights for evaluation.
```bash
./scripts/download_trained_model.sh
```

1. Evaluate the trained model performance.
```bash
./scripts/eval_MUSIC.sh
```

## Reference
If you use the code or dataset from the project, please cite:
```bibtex
    @InProceedings{Zhao_2018_ECCV,
        author = {Zhao, Hang and Gan, Chuang and Rouditchenko, Andrew and Vondrick, Carl and McDermott, Josh and Torralba, Antonio},
        title = {The Sound of Pixels},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        month = {September},
        year = {2018}
    }
```
