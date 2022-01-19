# QforCRNet
## Overview
This is a repository for paper "", which has been submitted to the IEEE for possible publication. Trained models and the test script are listed here as a validation of our work.
## Requirements
To use this project, you need to ensure the following requirements are installed.
- Python >= 3.8
- Pytorch == 1.6
- thop
## Project preparation
### A. Data Preparation
The dataset of channel state information (CSI) matrices is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── QCRNet  # The cloned CRNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── test.py
│   ├── run.bash
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── Checkpoints # The checkpoints folder
│   ├── in-cr4-6bit.pth
│   ├── ... 
...
```
## Results and reproduction
The main results reported in our paper are presented as follows. All the listed results can be found in Table1 and Table2 of our paper. They are achieved from our training scheme and quantization strategy.

Scenario | Compression Ratio | Quantization bit width | NMSE | SNR | Checkpoints
:--: | :--: | :--: | :--: | :--: | :--:
indoor | 1/4 | 6 | -26.99 |  | 5.12M | in_04.pth
indoor | 1/4 | 4 | -26.99 |  | 5.12M | in_04.pth
indoor | 1/8 | 6 | -16.01 |  | 4.07M | in_08.pth
indoor | 1/8 | 4 | -16.01 |  | 4.07M | in_08.pth
indoor | 1/16 | 6 | -11.35 |  | 3.55M | in_16.pth
indoor | 1/16 | 4 | -11.35 |  | 3.55M | in_16.pth
outdoor | 1/4 | 6 | -12.70 |  | 5.12M | out_04.pth
outdoor | 1/4 | 4 | -12.70 |  | 5.12M | out_04.pth
outdoor | 1/8 | 6 | -8.04 |  | 4.07M | out_08.pth
outdoor | 1/8 | 4 | -8.04 |  | 4.07M | out_08.pth
outdoor | 1/16 | 6 | -5.44 |  | 3.55M | out_16.pth
outdoor | 1/16 | 4 | -5.44 |  | 3.55M | out_16.pth

As aforementioned, we provide model checkpoints for all the results. Our code library supports easy inference. 

**To reproduce all these results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/CRNet/test.py \
  --data-dir '/home/COST2100' \  # the dataset path
  --scenario 'in' \ # the scenario: 'in' for indoor and 'out' for outdoor
  --pretrained './checkpoints/in_04' \ # the checkpoint path
  --cr 4 \  # the compression ratio: 4, 8, or 16
  --evaluate \
  --batch-size 200 \ 
  --workers 0 \
  --cpu \
```

## Acknowledgement

Thank zhilin for his open source code CRNet, you can find his work named CRNet in [Github-CRNet](https://github.com/Kylin9511/CRNet). Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 

