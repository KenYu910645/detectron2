# Panoptic-Depth Color Map for Combination of Depth and Image Segmentation

Jia-Quan Yu, Soo-Chang Pei

[[`arXiv`](https://arxiv.org/abs/1911.10194)]

<div align="center">
  <img src="https://github.com/KenYu910645/detectron2/tree/main/projects/Panoptic-DepthLab/picture/architecture.png"/>
</div><br/>

## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).
To use cityscapes, prepare data follow the [tutorial](https://detectron2.readthedocs.io/tutorials/builtin_datasets.html#expected-dataset-structure-for-cityscapes).

## Environment
OS: Ubuntu 20.04.6 LTS \
GPU: NVIDIA TITAN RTX x2 \
python     : 3.10.12 \
pytorch    : 2.0.1 \
cuda1      : 11.7 \
torchvision: 0.15.2 

## Training

To train a model with 2 GPUs run:
```bash
python train_net.py \
--config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
--num-gpus 2 \
MODEL.WEIGHTS model_final_bd324a.pkl
--resume
```
## Inference
```bash
export CUDA_VISIBLE_DEVICES=0
python train_net.py \
--config-file configs/Cityscapes-PanopticSegmentation/panoptic_depthlab.yaml \
--eval-only MODEL.WEIGHTS output_depthlab_L1loss/<the_ckpt_you_want_to_use>.pth
```

## Demo Result

<div align="center">
  <img src="https://github.com/KenYu910645/detectron2/tree/main/projects/Panoptic-DepthLab/picture/frankfurt_000000_000576_leftImg8bit_depthlab.png"/>
</div><br/>

<div align="center">
  <img src="https://github.com/KenYu910645/detectron2/tree/main/projects/Panoptic-DepthLab/picture/munster_000156_000019_leftImg8bit_depthlab.png"/>
</div><br/>

<div align="center">
  <img src="https://github.com/KenYu910645/detectron2/tree/main/projects/Panoptic-DepthLab/picture/munster_000167_000019_leftImg8bit_depthlab.png"/>
</div><br/>