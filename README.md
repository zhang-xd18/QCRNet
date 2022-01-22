# QCRNet
## Overview
This is a Pytorch implementation for paper [A Distribution Adapter for Quantization in Deep
Learning-Based Massive MIMO CSI Feedback](), which has been submitted to the IEEE for possible publication. The test script and trained models are listed here and the key results can be reproduced as a validation of our work .
## Requirements
To use this project, you need to ensure the following requirements are installed.
- Python >= 3.8
- Pytorch == 1.6
- thop
## Project preparation
### A. Data Preparation
The dataset of channel state information (CSI) matrices is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

### B. Checkpoints Downloading
The checkpoints of our proposed method can be downloaded from [Baidu Netdisk]() (passwd: he1o) or [Google Drive]().

### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── QCRNet  # The cloned QCRNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── test_QCRNet.py
├── COST2100  # COST2100 dataset downloaded following section A
│   ├── DATA_Htestin.mat
│   ├── ...
├── Checkpoints # The checkpoints folder downloaded following section B
│   ├── in-cr4-6bit.pth
│   ├── ... 
├── run.bash # The bash script
...
```
## Results and reproduction
The main results reported in our paper are presented as follows. All the listed results can be found in Table1 and Table2 of our paper. They are achieved from our proposed training scheme and quantization strategy.

Scenario | Compression Ratio | Quantization bit width | NMSE(dB) | SNR(dB) | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--:
indoor | 4 | 6 | -25.25 | 47.14 | in-cr4-6bit.pth
indoor | 4 | 4 | -18.63 | 30.47 | in-cr4-4bit.pth
indoor | 8 | 6 | -15.38 | 46.06 | in-cr8-6bit.pth
indoor | 8 | 4 | -13.95 | 33.14 | in-cr8-4bit.pth
indoor | 16 | 6 | -10.65 | 44.37 | in-cr16-6bit.pth
indoor | 16 | 4 | -10.56 | 31.64 | in-cr16-4bit.pth
outdoor | 4 | 6 | -11.79 | 49.28 | out-cr4-6bit.pth
outdoor | 4 | 4 | -11.69 | 28.46 | out-cr4-4bit.pth
outdoor | 8 | 6 | -8.223 | 44.70 | out-cr8-6bit.pth
outdoor | 8 | 4 | -8.110 | 29.26 | out-cr8-4bit.pth
outdoor | 16 | 6 | -5.437 | 41.56 | out-cr16-6bit.pth
outdoor | 16 | 4 | -5.287 | 26.11 | out-cr16-4bit.pth

As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

**To reproduce all these results, you need to download the given dataset and checkpoints. Also, you should arrange your projects tree as instructed.** An example of **run.sh** is shown as follows.

``` bash
python ./QCRNet/test_QCRNet.py \
  --data-dir './COST2100' \ # path to the COST2100 dataset
  --scenario 'in' \ # chosen from ['in','out'] for indoor or outdoor scenarios respectively
  --pretrained './checkpoints/in-cr4-6bit.pth' \  # path to the pretrained checkpoint
  --cr 4 \  # chosen from [4, 8, 16]
  --nbit 6 \  # chosen from [6, 4]
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cpu
```

  
## Acknowledgement

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thank zhilin for his work named [CRNet](https://ieeexplore.ieee.org/document/9149229) and you can refer to it if you are instereted in the network details. 

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset.

