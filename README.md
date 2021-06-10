# Semi-supervised Long-tailed Recognition using Alternate Sampling

## Overview
This is the author's pytorch implementation for the paper "Semi-supervised Long-tailed Recognition using Alternate Sampling". This will reproduce the alternate learning performance on ImageNet-SSLT with ResNet18. One can easily change the netwokrs or datasets by manipulating the args.

The model is designed to train on eight (8) Titan Xp (12GB memory each). Please adjust the batch size (or even learning rate) accordingly, if the GPU setting is different.

## Requirements
* [Python](https://python.org/) (version 3.7.6 tested)
* [PyTorch](https://pytorch.org/) (version 1.5.1 tested)

## Data Preparation
- First, please download the [ImageNet_2012](http://image-net.org/index).

- Next, change the `/your/ImageNet/path` in `CB_ResNet18.py`, and `alternate_imagenet.py` accordingly.

- The data splits are provided in the codes.

## Getting Started (Training & Testing)
- pre-training:
```
python init_train.py
```
- alternate training:
```
python alternate.py
```
