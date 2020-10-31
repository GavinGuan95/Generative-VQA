# Generative VQA

This repository contains implementation for the paper Generative Visual Question Answering using Cross-Modal Visual Linguistic Embeddings.  

## Introduction
Visual Question Answering (VQA) has usually been treated as an answer selection task, i.e. given a set of pre-defined answers, the model is required to select the best answer.
In this work, we frame VQA as an answering generation task: there is no pre-defined answer list, the model has to generate the full answer by itself.
This repository contains the source code for the VL-BERT + Meshed Decoder solution. The VL-BERT learns representations combining information from the input image and question.
The meshed decoder performs generation of the output sequence one by one.

## Prepare

### Environment
* Ubuntu 16.04, CUDA 9.0, GCC 4.9.4
* Python 3.6.x
    ```bash
    # We recommend you to use Anaconda/Miniconda to create a conda environment
    conda create -n vl-bert python=3.6 pip
    conda activate vl-bert
    ```
* PyTorch 1.0.0 or 1.1.0
    ```bash
    conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
    ```
* Apex (optional, for speed-up and fp16 training)
    ```bash
    git clone https://github.com/jackroos/apex
    cd ./apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./  
    ```
* Other requirements:
    ```bash
    pip install Cython
    pip install -r requirements.txt
    ```
* Compile
    ```bash
    ./scripts/init.sh
    ```

### Data

See [PREPARE_DATA.md](data/PREPARE_DATA.md).

### Pre-trained Models

See [PREPARE_PRETRAINED_MODELS.md](model/pretrained_model/PREPARE_PRETRAINED_MODELS.md).

## Training and Testing

To train the model, run:
```
cd vqa
./train_end2end.py --cfg ./cfgs/vqa/base_4x16G_fp32.yaml --model-dir ./ckpts_decoder_with_proper_acc2
```

To generate the result shown in the paper, run:
```
cd vqa
./test.py --cfg ./cfgs/vqa/base_4x16G_fp32.yaml --ckpt ./ckpts/output/vl-bert/vqa/base_4x16G_fp32/train2014_train/vl-bert_base_res101_vqa-best.model --bs 1 --gpus 0 --model-dir ./model/pretrained_model --result-path ./result --result-name result.txt
Following is a more concrete example:
```

## Acknowledgements
This code adapt previous works on:
* [VL-BERT](https://github.com/jackroos/VL-BERT) 
* [Meshed-Memory-Transformer](https://github.com/aimagelab/meshed-memory-transformer)
