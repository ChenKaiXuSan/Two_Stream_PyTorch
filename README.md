<div align="center">

# Two Stream 3D CNN for Video Classification

</div>

## Description

📓 This project made with the PyTorch, PyTorch Lightning, PyTorch Video.

This project implements the task of classifying different medical diseases.

We use a two stream based method combine the 3D CNN network for video classification.

The whole procedure is divided into two steps:  

1. using the detection method to extract the character-centered region and save it as a video.
2. a hidden-in RAFT based method to extract the optical flow of the corresponding images.
3. use the RGB images with the corresponding optical flow feed into a 3D CNN based network for training.

Detailed comments are written for most of the methods and classes.
Have a nice code. 😄

## How to run  

First, install dependencies

```bash
# clone project   
git clone https://github.com/ChenKaiXuSan/Two_Stream_PyTorch.git

# install project   
cd Two_Stream_PyTorch
pip install -e .   
pip install -r requirements.txt

```

Next, navigate to any file and run it.  

```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python main.py --[some paramer]
```

## About the lib  

stop building wheels. 🛑

### PyTorch Lightning  

[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. Lightning evolves with you as your projects go from idea to paper/production.

### PyTorch Video  

[link](https://pytorchvideo.org/)
A deep learning library for video understanding research.

### detectron2

[Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.

### Torch Metrics

[TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) is a collection of 80+ PyTorch metrics implementations and an easy-to-use API to create custom metrics.
