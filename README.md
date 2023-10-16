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

| Architecture | Trick         | Class Representation | Dim | Top1      | Top5      | Config                                                                    | Log                                                                                          | Checkpoint                                                                                    |
| ------------ | ------------- | -------------------- | --- | --------- | --------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ResNet50-D   |               | Vector               | -   | 78.05     | 93.90     | [config](configs/resnet/resnet50d_8xb32_coslr_in1k.py)                    | [log](https://drive.google.com/file/d/19NMZs1dlCshW1_8PJx7s200xl_RzoV-j/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1Co5FJYqDcqeEYw_ORTOrRxg3oh2acfRI/view?usp=drive_link) |
| ResNet50-D   |               | GCR                  | 1   | 78.42     | 94.14     | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr1.py)               | [log](https://drive.google.com/file/d/1e0sJRdtMZOEgM9AfENNsicHhtjMawgw7/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1m1F9TmOQjmKli-_U5ifSxVoiH8R0tHK1/view?usp=drive_link) |
| ResNet50-D   |               | GCR                  | 4   | 78.68     | 94.32     | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr4.py)               | [log](https://drive.google.com/file/d/12v7UZWrlh3ClIJmz6ysGGMN5TTg50XRI/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1tIXkD9zB7y4Ni3RZYNKgQTZw_PCZkCJV/view?usp=drive_link) |
| ResNet50-D   |               | GCR                  | 8   | 79.26     | 94.44     | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr8.py)               | [log](https://drive.google.com/file/d/18Hns24rv66163uPKiZZGnoEVal4An7Lc/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1FpuEpzqf-v3c8rsdFX882mKXLY07sXwc/view?usp=drive_link) |
| ResNet50-D   |               | GCR                  | 16  | 79.21     | 94.37     | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr16.py)              | [log](https://drive.google.com/file/d/1kodUwqZuACdi259DBUVhYlGyMpA-FsVb/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1N9l7Lb-gSMOK-1JzUqa0wLgb81_iBCw1/view?usp=drive_link) |
| ResNet50-D   |               | GCR                  | 32  | 78.63     | 94.05     | [config](configs/resnet/resnet50d_8xb32_coslr_in1k_gcr32.py)              | [log](https://drive.google.com/file/d/1O0SeintbVELA9_-hIcHIJx930TdlHQ9S/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1Rq2ugIMOZbpDufpiKpewKR4xc9DTExu7/view?usp=drive_link) |
| ResNet50-D   | RSB-A1        | Vector               | -   | 80.53     | 94.98     | [config](configs/resnet/resnet50d_8xb256-rsb-a1-600e_in1k.py)             | [log](https://drive.google.com/file/d/1cNw6Kf_BbDn4M3dFfy2gANr3aZgUKEo_/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1ororxl3RMuYTB7TD83nb9Fyq5VuBI5Pq/view?usp=drive_link) |
| ResNet50-D   | RSB-A1        | GCR                  | 8   | 81.00     | 95.40     | [config](configs/resnet/resnet50d_8xb256-rsb-a1-600e_in1k_gcr8.py)        | [log](https://drive.google.com/file/d/1SjpT47lTpUPnI33xM9SC3SpC0s440twP/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1njtbpwfz1duOilapCcUnTpGX2eTuTy_U/view?usp=drive_link) |
| ResNet50-D   | RSB-A1 FixRes | GCR                  | 8   | **81.30** | **95.42** | [config](configs/resnet/resnet50d_8xb256-rsb-a1-600e_in1k_gcr8_fixres.py) | [log](https://drive.google.com/file/d/1QSXNiLUK-6VhrktW2hlkNcGS95AJ7RoZ/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/17hYOOMLrg8wOs6g1ehtfbQ_gMWsoYh9j/view?usp=drive_link) |
| ResNet50-D   | RSB-A2        | Vector               | -   | 80.29     | 94.86     | [config](configs/resnet/resnet50d_8xb256-rsb-a2-300e_in1k.py)             | [log](https://drive.google.com/file/d/1qaVXUYU7SYhuk5ZeRAblGpVC38XvSv8M/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1LOVRzzP3BfhRvdDgSUgTJ3So9NxllX3N/view?usp=drive_link) |
| ResNet50-D   | RSB-A2        | GCR                  | 8   | 80.75     | 95.24     | [config](configs/resnet/resnet50d_8xb256-rsb-a2-300e_in1k_gcr8.py)        | [log](https://drive.google.com/file/d/14BXtMZIM1SEJLdQzRveloE0nRIMcuxH6/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1Vm5xqOXzRXTK_4URXW8rbeevOHXgUOBP/view?usp=drive_link) |
| ResNet50-D   | RSB-A2 FixRes | GCR                  | 8   | 81.04     | 95.46     | [config](configs/resnet/resnet50d_8xb256-rsb-a2-300e_in1k_gcr8_fixres.py) | [log](https://drive.google.com/file/d/17Wyg0r5_a8ENpoPMQOdtXUxUxPoc6P9L/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1RRHAGOd14B7wQb5tcR5blxcm5EBk4wqw/view?usp=drive_link) |
| ResNet50-D   | RSB-A3        | Vector               | -   | 79.36     | 94.47     | [config](configs/resnet/resnet50d_8xb256-rsb-a3-100e_in1k.py)             | [log](https://drive.google.com/file/d/1pRxmvOcExofZlhL3ajB99Y7CtQiaC8gD/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1SXSfGEE_OOBDqMYXOFJOmJsj5nTe8Mpy/view?usp=drive_link) |
| ResNet50-D   | RSB-A3        | GCR                  | 8   | 79.64     | 94.85     | [config](configs/resnet/resnet50d_8xb256-rsb-a3-100e_in1k_gcr8.py)        | [log](https://drive.google.com/file/d/1x9BGutIOzdvZwvTxSwbu-e7GSWINDXCv/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1gD11fWCReLKU0hoqJzK-7oBZDjWA0S-U/view?usp=drive_link) |
| ResNet50-D   | RSB-A3 FixRes | GCR                  | 8   | 80.20     | 95.15     | [config](configs/resnet/resnet50d_8xb256-rsb-a3-100e_in1k_gcr8_fixres.py) | [log](https://drive.google.com/file/d/1rNwRPSFz_-FLOJxTLnUPSOvSqtrAFBm0/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1ZDXr5V0KIUEk64LrdwTaAvayw7dXD-VL/view?usp=drive_link) |
| ResNet101-D  |               | Vector               | -   | 79.31     | 94.67     | [config](configs/resnet/resnet101d_8xb32-coslr_in1k.py)                   | [log](https://drive.google.com/file/d/1f7si6980G9ni6mC8o2dU5n9yQ-yRtav3/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1RZNdlM00B1TWoOT3YpTD3_TZY0ZDjGdI/view?usp=drive_link) |
| ResNet101-D  |               | GCR                  | 8   | 80.24     | 94.95     | [config](configs/resnet/resnet101d_8xb32-coslr_in1k_gcr8.py)              | [log](https://drive.google.com/file/d/1PMEOvvh5z602T7HdwfPQCafkVjeXMceJ/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1SS3iyVko63lA6G4OTlDVgVWGmzTV2njF/view?usp=drive_link) |
| ResNet152-D  |               | Vector               | -   | 80.00     | 95.02     | [config](configs/resnet/resnet152d_8xb32-coslr_in1k.py)                   | [log](https://drive.google.com/file/d/1FWxJyDcEpnoRStN16n2GyIevEJ-hHkS1/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1wRIL19r_c8o_2eXZE2ARtEGf77KxhiVn/view?usp=drive_link) |
| ResNet152-D  |               | GCR                  | 8   | 80.44     | 95.21     | [config](configs/resnet/resnet152d_8xb32-coslr_in1k_gcr8.py)              | [log](https://drive.google.com/file/d/1e4UO1jtTViv7zww6wnR__hinpNSjPx0R/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1yFiJ9lqy9uD1HQ02MWzw_ZlXLiqjK4BY/view?usp=drive_link) |
| VGG13-BN     |               | Vector               | -   | 72.02     | 90.79     | [config](configs/vgg/vgg13bn_8xb32_in1k.py)                               | [log](https://drive.google.com/file/d/1SsLzmDmiQGaDToIYWk-9f8tOp2Mscu3R/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1AP2fI7nFGRZ6mU__E8N-MV5ecEZ1KW4o/view?usp=drive_link) |
| VGG13-BN     |               | GCR                  | 8   | 73.40     | 91.30     | [config](configs/vgg/vgg13bn_8xb32_in1k_gcr8.py)                          | [log](https://drive.google.com/file/d/1tt_iy_R2uSRXm2adPOsLC1uVbij4pEud/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/10EJP9eU-EoFxy11T_qOm8urUHj71BuFz/view?usp=drive_link) |
| Swin-T       |               | Vector               | -   | 81.06     | 95.51     | [config](configs/swin_transformer/swin-tiny_8xb128_in1k.py)               | [log](https://drive.google.com/file/d/1USGKrheQe6jK1xkFPgUt8M1DFpGc8FNr/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1xstvE_t89F_SNc_zDm2OxRJ-Irq5SLYN/view?usp=drive_link) |
| Swin-T       |               | GCR                  | 8   | 81.63     | 95.77     | [config](configs/swin_transformer/swin-tiny_8xb128_in1k_gcr8.py)          | [log](https://drive.google.com/file/d/1_mTgAMA95rkwMpzM0hjmM7qNN5V4CNFV/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1RIpMraINw6CjvUAzNiAJOm11xQq-SUSx/view?usp=drive_link) |
| Deit3-S      |               | Vector               | -   | 81.53     | 95.22     | [config](configs/deit3/deit3-small-p16_8xb256_in1k-224px.py)              | [log](https://drive.google.com/file/d/11FEoYMfzTxOXeFAubz-BsnEhv7mCrXYW/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1WFPbxiOOu380qsYQeNFtRSwVJV4eZR12/view?usp=drive_link) |
| Deit3-S      |               | GCR                  | 8   | 82.18     | 95.73     | [config](configs/deit3/deit3-small-p16_8xb256_in1k-224px_gcr8.py)         | [log](https://drive.google.com/file/d/1r-mJZ04DFcJwMfDij9fSR7eEh16cyZeK/view?usp=drive_link) | [ckpt](https://drive.google.com/file/d/1n-uHEWuiyJ1OLeuD555hfyJugKmXBYCx/view?usp=drive_link) |

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
