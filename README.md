# CLIPformer

CLIPFormer: Language-Driven Remote Sensing Change Detection with Context-Aware Prompts

## Introduction

In this work, we propose a new change detection framework, CLIPFormer, which leverages pretraining knowledge from CLIP and the Swin transformer. 
 
## Install
- First, you need to download mmsegmentation and install it on your server.
- Second, place backbone.py and csheadunet.py in the corresponding directory of mmsegmentation.
- Third, train according to the training strategy of mmsegmentation and the training parameters in our paper.

## Pretrained Weights of Backbones

[pretrain](https://download.openmmlab.com/mmsegmentation/)

## Data Preprocessing

Download the datasets from the official website and split them yourself.



**LEVIR-CD**
[LEVIR-CD]()

**LEVIR-CD+**
[LEVIR-CD+]()

**WHUCD**
[WHUCD]()

**CDD**
[CDD]()

**SYSU-CD**
[SYSU-CD]()


## Training

You can refer to **mmsegmentation document** (https://mmsegmentation.readthedocs.io/en/latest/index.html).


## Results and Logs for CLIPformer

### TABLE I
COMPARISONS OF DETECTION PERFORMANCE ON LEVIR-CD DATASET
| Model | Backbone | OA | IoU | F1 | Prec | Rec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BIT | Resnet18 | 98.92 | 80.66 | 89.29 | 90.56 |  88.06 | 

| Dataset | Crop Size | Lr Schd | mIoU | #params(Mb) | FLOPs(Gbps) | config | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Potsdam | 512x512 | 100K | 75.44 | 23.2 | 18.5 | [config](https://github.com/wujiang0156/MCAT-UNet/blob/main/config/segnext_csheadunet-s_1xb4-adamw-100k_potsdam-512x512.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20231219_003217.log)
| Vaihingen | 512x512 | 100K | 74.52 |  23.2 | 18.5 | [config](https://github.com/wujiang0156/MCAT-UNet/blob/main/config/segnext_csheadunet-s_1xb4-adamw-100k_vaihingen-512x512.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20240106_074357.log)
| LoveDa | 512x512 | 100K | 53.58 |  23.2 | 18.5 | [config](https://github.com/wujiang0156/MCAT-UNet/blob/main/config/segnext_csheadunet-s_1xb4-adamw-100k_loveda-512x512.py) | [github](https://github.com/wujiang0156/MCAT-UNet/releases/download/LOG/20231226_030251.log)
## Inference on High-resolution remote sensing image

<div>
<img src="fig 6.jpg" width="100%"/>
<img src="fig 7.jpg" width="100%"/>
<img src="fig 8.jpg" width="100%"/>
<img src="fig 9.jpg" width="100%"/>
</div>

## Acknowledgement

 Many thanks the following projects's contributions to **MACT-UNet**.
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
