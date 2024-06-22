# CLIPformer

CLIPFormer: Language-Driven Remote Sensing Change Detection with Context-Aware Prompts

## Introduction

In this work, we propose a new change detection framework, CLIPFormer, which leverages pretraining knowledge from CLIP and the Swin transformer. 
 
## Install
- First, you need to download mmsegmentation and install it on your server.
- Second, place clipformer.py, swinclip and cswin_text_head.py in the corresponding directory of mmsegmentation.
- Third, train according to the training strategy of mmsegmentation and the training parameters in our paper.

## Pretrained Weights of Backbones

[CLIP-pretrain](https://github.com/OpenAI/CLIP)
[Swin-Trnasformer-pretrain](https://download.openmmlab.com/mmsegmentation/)


## Data Preprocessing

Download the datasets from the official website and split them yourself.



**LEVIR-CD**
[LEVIR-CD](https://justchenhao.github.io/LEVIR/)

**LEVIR-CD+**
[LEVIR-CD+](https://github.com/S2Looking/Dataset)

**WHUCD**
[WHUCD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

**CDD**
[CDD](https://paperswithcode.com/sota/change-detection-for-remote-sensing-images-on)

**SYSU-CD**
[SYSU-CD](https://github.com/liumency/SYSU-CD)


## Training

You can refer to **mmsegmentation document** (https://mmsegmentation.readthedocs.io/en/latest/index.html).


## Results and Logs for CLIPformer
Here we only present the test results of our model. For detailed test results, please refer to our paper.
**We will provide the log files later.**
### TABLE I
COMPARISONS OF DETECTION PERFORMANCE ON LEVIR-CD DATASET
| Model | Backbone | OA | IoU | F1 | Prec | Rec | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIPformer(ViT-B/16) | Swin-T | 99.22 | 85.60 | 92.24 | 93.60 |  90.92 | [log](https://github.com/wujiang0156/CLIPformer/blob/main/log/20240413_094938.log) |

### TABLE II
COMPARISONS OF DETECTION PERFORMANCE ON LEVIR-CD+ DATASET
| Model | Backbone | OA | IoU | F1 | Prec | Rec | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIPformer(ViT-B/16) | Swin-T | 98.87 | 76.81 | 86.89 | 88.51 |  85.32 | [log](https://github.com/wujiang0156/CLIPformer) | 

### TABLE III
COMPARISONS OF DETECTION PERFORMANCE ON WHUCD DATASET
| Model | Backbone | OA | IoU | F1 | Prec | Rec | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIPformer(ViT-B/16) | Swin-T | 99.54 | 89.55 | 94.49 | 96.38 |  92.66 | [log](https://github.com/wujiang0156/CLIPformer) | 

### TABLE IV
COMPARISONS OF DETECTION PERFORMANCE ON CDD DATASET
| Model | Backbone | OA | IoU | F1 | Prec | Rec | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIPformer(ViT-B/16) | Swin-T | 99.33 | 94.51 | 97.18 | 97.03 |  97.32 | [log](https://github.com/wujiang0156/CLIPformer) | 

### TABLE V
COMPARISONS OF DETECTION PERFORMANCE ON SYSU-CD DATASET.
| Model | Backbone | OA | IoU | F1 | Prec | Rec | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIPformer(ViT-B/16) | Swin-T | 99.62 | 71.77 | 83.57 | 88.02 |  79.54 | [log](https://github.com/wujiang0156/CLIPformer) | 

## Visualization on remote sensing change detection datasets
For detailed qualitative analysis and visualization results on the LEVIR-CD, LEVIR-CD+, WHUCD, CDD, and SYSU-CD datasets, please refer to our paper.
<div>
<img src="XXX.jpg" width="100%"/>
<img src="XXX.jpg" width="100%"/>
<img src="XXX.jpg" width="100%"/>
<img src="XXX.jpg" width="100%"/>
 <img src="XXX.jpg" width="100%"/>
</div>

## Acknowledgement

 Many thanks the following projects's contributions to **MACT-UNet**.
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [DenseCLIP](https://github.com/raoyongming/DenseCLIP)
- [CLIP](https://github.com/OpenAI/CLIP)
- [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)
