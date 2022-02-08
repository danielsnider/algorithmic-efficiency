## 1. Overview

Speech recognition accepts raw audio samples and produces a corresponding text
transcription. We train a CNN LSTM model on LibriSpeech clean 100 hour dataset
and evaluate it on Librispeech test clean dataset. We get Character Error Rate
(CER) 0.0995 after 52760 steps or 52825.76 seconds on a 8 V-100 GPU machine
(1.00 step/second).

## 2. Download and preprocess dataset

```
mkdir data_dir
cd data_dir

download_data.sh

cd ..
mkdir work_dir
cd work_dir

python prepare_data.py data_dir/LibriSpeech
```

The raw dataset is under `data_dir` and the preprocessed dataset is under
`work_dir`.

## 3. Reference

https://github.com/lsari/librispeech_100

