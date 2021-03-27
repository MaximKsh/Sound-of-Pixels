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
* MUSIC: [https://disk.yandex.ru/d/lj-wdG4CXz8fUw](https://disk.yandex.ru/d/lj-wdG4CXz8fUw)
* VGGSound: [https://disk.yandex.ru/d/u4OooLD30Ldzbw](https://disk.yandex.ru/d/u4OooLD30Ldzbw)

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
* MUSIC:
```
python3 main.py train --config music.json 
```
* VGGSound 
```
python3 main.py train --config vggsound.json
```

## Evaluation
* MUSIC:
```
python3 main.py eval --config music.json 
```
* VGGSound 
```
python3 main.py eval --config vggsound.json
```

## Inference
* MUSIC:
```
python3 main.py regions --config music.json 
```
* VGGSound 
```
python3 main.py regions --config vggsound.json
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
