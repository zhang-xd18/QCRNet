# QCRNet
## Overview
This is a Pytorch implementation for paper [Quantization Adaptor for Bit-Level Deep
Learning-Based Massive MIMO CSI Feedback](https://arxiv.org/abs/2211.02937), which has been submitted to the IEEE for possible publication. The test script and trained models are listed here and the key results can be reproduced as a validation of our work .
## Requirements
To use this project, you need to ensure the following requirements are installed.
- Python >= 3.8
- Pytorch == 1.6
- thop
## Project preparation
### A. Data Preparation
We consider both the COST2100 and the CDL channel models in this work.
COST2100 dataset is generated referring to this [paper](https://ieeexplore.ieee.org/document/6393523). Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset which is easier to use for the CSI feedback task. You can download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

The CDL dataset is generated referring to the Wireless-Intelligence libirary from OPPO. You can download it from their official website [Wireless-Intelligence](https://wireless-intelligence.com/#/home). The parameter settings are described in detail in our paper. We also provide download links of [Google Drive](https://drive.google.com/drive/folders/1vci-FVjjidIQxKpsc7d0pAu0AIiPwxom?usp=share_link) or [Baidu Netdisk](https://pan.baidu.com/s/1YHVeI4M8OFQmHlDqCZDSjw) (passwd:4gbc).


### B. Checkpoints Downloading
The checkpoints of our proposed method can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1YWX8zb52GNSV8phSPO-3eA) (passwd: yn5b) or [Google Drive](https://drive.google.com/drive/folders/1vci-FVjjidIQxKpsc7d0pAu0AIiPwxom?usp=sharing).

## Performance Reproduction
### A. Network-based Adaptor
#### Project tree arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CRNet_NA  # The cloned QCRNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── Dataset  # Datasets downloaded following Data Preparation
│   ├── DATA_Htestin.mat
│   ├── ...
├── checkpoint_NA # The checkpoints folder downloaded following Checkpoints Downloading
│   ├── in_4_6bit.pth
│   ├── ... 
├── run.sh # The bash script
...
```
#### Results and reproduction
The main results reported in our paper are presented as follows. All the listed results can be found in Table I and Table II of our paper. They are achieved from our proposed training scheme and quantization strategy.

Scenario | Compression Ratio | Quantization bit width | NMSE(dB) | SNR(dB) | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--:
indoor | 4 | 6 | -25.82 | 47.11 | in_4_6bit.pth
indoor | 4 | 4 | -19.94 | 30.76 | in_4_4bit.pth
indoor | 8 | 6 | -15.58 | 46.19 | in_8_6bit.pth
indoor | 8 | 4 | -14.32 | 29.71 | in_8_4bit.pth
indoor | 16 | 6 | -11.68 | 43.62 | in_16_6bit.pth
indoor | 16 | 4 | -10.55 | 29.02 | in_16_4bit.pth
outdoor | 4 | 6 | -12.70 | 39.06 | out_4_6bit.pth
outdoor | 4 | 4 | -12.13 | 27.16 | out_4_4bit.pth
outdoor | 8 | 6 | -8.352 | 36.73 | out_8_6bit.pth
outdoor | 8 | 4 | -8.102 | 27.00 | out_8_4bit.pth
outdoor | 16 | 6 | -5.513 | 35.50 | out_16_6bit.pth
outdoor | 16 | 4 | -5.359 | 26.65 | out_16_4bit.pth

As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

To reproduce all these results, you need to download the given dataset and corresponding checkpoints. Also, you should arrange your projects tree as instructed. An example of `run.sh` is shown as follows.

``` bash
python ./CRNet_NA/main.py \
  --data-dir './Dataset' \ # path to the dataset
  --scenario 'in' \ # chosen from ['in','out'] for indoor or outdoor scenarios respectively.
  --pretrained './checkpoint_NA/in_4_6bit.pth' \  # path to the pretrained checkpoint
  --cr 4 \  # compression ratio
  --nbit 6 \  # quantization bit width, chosen from [6, 4]
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cpu
```

### B. Expert knowledge-based Adaptor
#### Project tree arrangement

We recommend you to arrange the project tree as follows.

```
home
├── CRNet_LA  # The cloned QCRNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── Dataset  # Datasets downloaded following Data Preparation
│   ├── DATA_Htestin.mat
│   ├── ...
├── checkpoint_LA # The checkpoints folder downloaded following Checkpoints Downloading
│   ├── in_4_6bit.pth
│   ├── ... 
├── run.sh # The bash script
...
```
#### Results
We test the performance of the L1 adaptor on both the COST2100 and the CDL channel datasets. The results are listed in the following. 
##### Performance on the COST2100 dataset
All the listed results can be found in Table I and Table II of our paper. 

Scenario | Compression Ratio | Quantization bit width | NMSE(dB) | SNR(dB) | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--:
indoor | 4 | 6 | -25.25 | 47.14 | in_4_6bit.pth
indoor | 4 | 4 | -18.71 | 30.58 | in_4_4bit.pth
indoor | 8 | 6 | -15.38 | 46.06 | in_8_6bit.pth
indoor | 8 | 4 | -13.95 | 33.14 | in_8_4bit.pth
indoor | 16 | 6 | -11.61 | 44.42 | in_16_6bit.pth
indoor | 16 | 4 | -10.48 | 31.66 | in_16_4bit.pth
outdoor | 4 | 6 | -11.79 | 49.28 | out_4_6bit.pth
outdoor | 4 | 4 | -11.69 | 28.46 | out_4_4bit.pth
outdoor | 8 | 6 | -8.223 | 44.70 | out_8_6bit.pth
outdoor | 8 | 4 | -8.045 | 29.90 | out_8_4bit.pth
outdoor | 16 | 6 | -5.437 | 41.56 | out_16_6bit.pth
outdoor | 16 | 4 | -5.287 | 26.11 | out_16_4bit.pth

##### Performance on the CDL dataset
All the listed results can be found in Table V of our paper. 

Delay Spread(ns) | Compression Ratio | Quantization bit width | NMSE(dB) | SNR(dB) | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--:
30 | 16 | 6 | -30.89 | 45.20 | A30_16_6bit.pth
30 | 16 | 4 | -22.59 | 29.26 | A30_16_4bit.pth
30 | 64 | 6 | -21.80 | 44.71 | A30_64_6bit.pth
30 | 64 | 4 | -16.29 | 29.65 | A30_64_4bit.pth
30 | 128 | 6 | -15.46 | 44.32 | A30_128_6bit.pth
30 | 128 | 4 | -12.64 | 29.72 | A30_128_4bit.pth
300 | 16 | 6 | -27.68 | 47.50 | A300_16_6bit.pth
300 | 16 | 4 | -18.98 | 30.29 | A300_16_4bit.pth
300 | 64 | 6 | -13.86 | 45.99 | A300_64_6bit.pth
300 | 64 | 4 | -12.35 | 31.32 | A300_64_4bit.pth
300 | 128 | 6 | -10.33 | 43.53 | A300_128_6bit.pth
300 | 128 | 4 | -9.700 | 29.33 | A300_128_4bit.pth

As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

### Reproduction
To reproduce all these results, you need to download the given dataset and checkpoints. Also, you should arrange your projects tree as instructed. An example of `run.sh` is shown as follows.

``` bash
python ./CRNet_LA/main.py \
  --data-dir './Dataset' \ # path to the dataset
  --scenario 'in' \ # chosen from ['in', 'out', 'A30', 'A300']. 'in' or 'out' for indoor or outdoor scenarios of the COST2100 dataset. 'A30' or 'A300' for the CDL dataset with delay spread set to 30 or 300 ns.
  --pretrained './checkpoint_LA/in_4_6bit.pth' \  # path to the pretrained checkpoint
  --cr 4 \  # compression ratio
  --nbit 6 \  # quantization bit width, chosen from [6, 4]
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cpu
```


## Acknowledgement

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thank zhilin for his work named [CRNet](https://ieeexplore.ieee.org/document/9149229) and you can refer to it if you are instereted in the details. 

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset.

Thank OPPO for providing the open wireless communication dataset Wireless-Intelligence.

