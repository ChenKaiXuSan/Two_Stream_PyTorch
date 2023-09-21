Why we need optical flow in human gait recognition?
===
A disscussion about the Optical Flow of the gait image in time series.
---

## Introduction

In human walking, there have two ways to capture the motion of the human gait.
1. 3D CNNs
2. Two-Stream CNNs (single RGB image + optical flow)

Optical Flow is a method to capture the motion of the human gait, and it can capture the motion information from two continuous frames, in time series.

Why we need optical flow in human gait recognition?
In my opinion, for the spinal disease recognition, we need to capture the motion information, not the static information.
Because the analysis of the spinal disease is based on the motion information, not the static information.

So, one way is to use the optical flow to capture the motion information, in different frames, in time series.

Why 3D CNNs is not enough?

3D CNNs is a method to capture the motion information, but it can only capture the motion information in one frame, not the motion information in different frames, in time series.

## Optical Flow

In general, the optical flow is a 2D vector field, which can be used to describe the motion information in two continuous frames, in time series.
It can calculate the motion information in two continuous frames, in pixel bias level.
But, if we do not use the continuous frames, instead, we use the uncontinuous frames, how the optical flow results will be?

The next figure is a color wheel for the optical flow.
The legend for optical visualization shows how direction is mapped to color and magnitude is mapped to saturation.

![color wheel](https://hci.iwr.uni-heidelberg.de/sites/default/files/node/images/687377970/legend_flow.png)

Figure 1. The color wheel for the optical flow. Reference from [^1].

## How the optical flow results will be?

Here, we visualize the optical flow results, in different type.

### Flow to image

![vis_flow](./imgs/optical_flow/visualization_OF_clean_background.png)

<center>Figure 1. The visualization of the optical flow results. Transform the optical flow to RGB image.</center>

### Different dimension of the optical flow



## What is the different between continuous and uncontinuous frames for optical flow predict?

In this section, we will discuss the different between continuous and uncontinuous frames for optical flow predict.
As we konw, the Optical Flow predict is based on the continuous frames, in time series.
But the question is, if we use the uncontinuous frames, how the optical flow results will be?
Is it perform well as the continuous frames?

![subsample](./imgs/optical_flow/uniform_temporal_subsample.png)
<center>Figure 2. The uniform temporal subsample to extract the uncontinuous frame from raw FPS.</center>

Figure 2 is the frame subsample method used in training.
For train model, we uniformly subsample the frames from the raw FPS.
But the problem is that, when use the subsample method to extracted frames, it will be uncontinuous frames.

![compare_frame](./imgs/optical_flow/compare_frame.png)

<center>Figure 3. The predicted Optical Flow results, compared between continuous and uncontinuous frames.</center>

Figure 3 show the predicted Optical Flow results, compared between continuous and uncontinuous frames.

- The upper of the figure is the uncontinuous frames, it uniformly subsample 8 frames from the raw 30 FPS,
- The lower of the figure is the continuous frames, it include all the 30 frames from the raw 30 FPS.

The right cloumn of the figure is the predicted Optical Flow results, with continuous and uncontinuous frames.


## How to get a clean Optical Flow edge results?

As we know, the Optical Flow method calculate the motion information in pixel bias level.
But, even if we use the segmentated frame (which means have clean body edge), the Optical Flow results is still noisy, in background.
The reason we think is that, the two continuous frames have different pixel bias in background, so the Optical Flow results will be noisy, in background.

In this research, we need to find a way to get a clean Optical Flow edge results, which means the Optical Flow results only contain the human part motion change information, not the background.

So, how to get a clean Optical Flow edge results?

A simple enough way is to use the mask to get the clean optical flow body edge, it means we can overlay the mask on the optical flow results, and then we can get the clean optical flow body edge.

## Process

For visualize the importance of the optical flow in human gait, we simulated a neural network processing flow which contains the different parts.

```mermaid
graph LR
    A[Input] --> B[Segmentation]
    B --> C[Optical Flow]
    C --> B
    B --> D[Optical Flow feature]
    D --> E[3D CNN]
    E --> F[Batch Normalization 3D]
    F --> G[Max Pooling 3D]
```

## Dose the optical flow method also need clean background?

In this section, we will discuss the different between clean background and noisy background for optical flow predict.

![compare_background](./imgs/optical_flow/compare_OF.png)

### OF predicted with clean background

### OF predicted without clean background


<!-- # todo: add the visualization of the optical flow -->
Here we list the parameters of different layers.
(The code is implemented by PyTorch)

```python
conv3d = Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
bn3d = BatchNorm3d(num_features=1)
maxpool3d = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
```

### 3D CNN

The input channels for RGB image is set to 3, and the output channels is set to 1, for easy to visualize.
The kernel size is set to (1, 7, 7), which means the kernel size in time dimension is 1, and the kernel size in height and width dimension is 7.
The stride is set to (1, 2, 2), which means the stride in time dimension is 1, and the stride in height and width dimension is 2.
The padding is set to (0, 1, 1), which means the padding in time dimension is 0, and the padding in height and width dimension is 1.

The formula for calculating the output size is:

$$
D_{out} = \left[ \frac{(D_{in} + 2 \times padding[0] - dilation[0] \times (kernel\_size[0] - 1) - 1)}{stride[0]} +1 \right]
$$
$$
H_{out} = \left[ \frac{(H_{in} + 2 \times padding[1] - dilation[1] \times (kernel\_size[1] - 1) - 1)}{stride[1]} +1 \right]
$$
$$
W_{out} = \left[ \frac{(W_{in} + 2 \times padding[2] - dilation[2] \times (kernel\_size[2] - 1) - 1)}{stride[2]} +1 \right]
$$

### Batch Normalization 3D

Batch Normalization 3D is a 3D version of Batch Normalization.
The batch normalization is used to normalize the input data, and make the input data have the same distribution.
The num_features is set to 1, which means the input channels is 1.
(The num_features is equal to the output channels of the previous layer.)

### Max Pooling 3D

The motivation of using Pooling 3D layer is to reduce the size of the feature map.
The Pooling layer is always used after the Convolution layer, on other word, the Pooling layer is set after the CNN layer.

The kernel size is set to (1, 3, 3), which means the kernel size in time dimension is (1, 2, 2), and the kernel size in height and width dimension is (0, 1, 1).

> [!Note] 
> The Pooling layer do not include the learnable parameters. It just decrease the size of the feature map.

## Visualization

![segmentation](./imgs/segmentation/seg.png)

<center>Figure 1. The visualization of segmentation results.</center>

![no_seg](./imgs/segmentation/no_seg.png)

<center>Figure 2. The visualization of no segmentation results.</center>

Figure 1 is the visualization of the segmentation results, and Figure 2 is the visualization of the no segmentation results.

In Figure 1, we can see that the segmentation results given a clean body edge, we think the segmentation results can make the CNNs pay more attention to the human part, not the background.

In Figure 2, we can see that the no segmentation results given a noisy body edge, and the background is changed frequently, because the background is also moving.
We think the no segmentation results can make the CNNs pay more attention to the background, not the human part.

## Conclusion

In summary, in this visualization, we compare the segmentation and no segmentation results, in human gait.

When training, we need CNNs to pay more attention to the human part, not the background.
One way is to use the segmentation to get the clean body edge, and then use the clean body edge to train the model.
On the other word, clean body edge can only contain the human part information, will drop the irrelevant information, such as the background.

We hope this visualization can help you understand the importance of the segmentation in human gait.

## Reference

[^1]: [A Toolbox to Visualize Dense Image Correspondences](https://hci.iwr.uni-heidelberg.de/content/toolbox-visualize-dense-image-correspondences)
[^2]: [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)