# Grassmann Class Representation

[![🦢 - Paper](https://img.shields.io/badge/🦢-Paper-red)](https://arxiv.org/pdf/2308.01547)
[![🌊 - Poster](https://img.shields.io/badge/🌊-Poster-blue)](./resource/gcr-iccv2023-poster.pdf)

Official code for "Get the Best of Both Worlds: Improving Accuracy and Transferability by Grassmann Class Representation (ICCV 2023)"

https://github.com/innerlee/gcr/assets/9464825/37708eb8-740c-46df-ad24-7bcf6ef592c9

![Grassmann class representation](./resource/pipeline_gcr.jpg)

## Installation

Run

```bash
python setup.py develop
```

### Requirements

- pytorch >= 1.13
- mmpretrain

### Prepare data

Put ImageNet1-K dataset on `data/imagenet` folder

## ImageNet1-K Pretrained Weights

| Architecture             | Class Representation | Dim | Top1  | Top5  | Config                                                       | Log                                                                                          | Checkpoint                                                                                    |
| ------------------------ | -------------------- | --- | ----- | ----- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ResNet50-D               | Vector               | -   | 78.05 | 93.90 | [config](configs/resnet/resnet50d_8xb32_coslr_in1k.py)       | [log](https://drive.google.com/file/d/19NMZs1dlCshW1_8PJx7s200xl_RzoV-j/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1Co5FJYqDcqeEYw_ORTOrRxg3oh2acfRI/view?usp=drive_link) |
| ResNet50-D               | GCR                  | 1   | 78.42 | 94.14 | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr1.py)  | [log](https://drive.google.com/file/d/1e0sJRdtMZOEgM9AfENNsicHhtjMawgw7/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1m1F9TmOQjmKli-_U5ifSxVoiH8R0tHK1/view?usp=drive_link) |
| ResNet50-D               | GCR                  | 4   | 78.68 | 94.32 | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr4.py)  | [log](https://drive.google.com/file/d/12v7UZWrlh3ClIJmz6ysGGMN5TTg50XRI/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1tIXkD9zB7y4Ni3RZYNKgQTZw_PCZkCJV/view?usp=drive_link) |
| ResNet50-D               | GCR                  | 8   | 79.26 | 94.44 | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr8.py)  | [log](https://drive.google.com/file/d/18Hns24rv66163uPKiZZGnoEVal4An7Lc/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1FpuEpzqf-v3c8rsdFX882mKXLY07sXwc/view?usp=drive_link) |
| ResNet50-D               | GCR                  | 16  | 79.21 | 94.37 | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr16.py) | [log](https://drive.google.com/file/d/1kodUwqZuACdi259DBUVhYlGyMpA-FsVb/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1N9l7Lb-gSMOK-1JzUqa0wLgb81_iBCw1/view?usp=drive_link) |
| ResNet50-D               | GCR                  | 32  | 78.63 | 94.05 | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr32.py) | [log](https://drive.google.com/file/d/1O0SeintbVELA9_-hIcHIJx930TdlHQ9S/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1Rq2ugIMOZbpDufpiKpewKR4xc9DTExu7/view?usp=drive_link) |
<!-- | ResNet50-D RSB-A1        | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A1 FixRes | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A2        | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A2        | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A2 FixRes | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A3        | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A3        | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet50-D RSB-A3 FixRes | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet101-D              | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet101-D              | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet152-D              | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNet152-D              | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNetXt50               | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| ResNetXt50               | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| VGG13-BN                 | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| VGG13-BN                 | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| Swin-T                   | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| Swin-T                   | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               |
| Deit3-S                  | Vector               | -   |       |       |                                                              |                                                                                              |                                                                                               |
| Deit3-S                  | GCR                  | 8   |       |       |                                                              |                                                                                              |                                                                                               | -->

## Training

```bash
# train ResNet50d baseline with 8 gpus
./tools/dist_train.sh configs/resnet/resnet50d_8xb32-coslr_in1k.py 8

# train the gcr version
./tools/dist_train.sh configs/resnet/resnet50d_8xb32-coslr_in1k_gcr8.py 8
```

## Citation

```
@inproceedings{haoqi2023gcr,
title = {Get the Best of Both Worlds: Improving Accuracy and Transferability by Grassmann Class Representation},
author = {Wang, Haoqi and Li, Zhizhong and Zhang, Wayne},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
year = {2023}
}
```


## Related Projects

[ViM: Out-Of-Distribution with Virtual-logit Matching (CVPR 2022)](https://github.com/haoqiwang/vim)
