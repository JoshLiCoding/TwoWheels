# TwoWheels

TwoWheels is a video-based object detection project, primarily for identifying cyclists. I aim to fine-tune two architectures - Faster R-CNN and SSD - and compare their results visually.

## Data

Images and labels were retrieved from the Tsinghua-Daimler Cyclist Benchmark (TDCB)'s validation dataset (see their [website](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/Tsinghua-Daimler_Cyclist_Detec/tsinghua-daimler_cyclist_detec.html)). The original bounding boxes were labelled with 6 categories: "pedestrian", "cyclist", "motorcyclist", "tricyclist", "wheelchairuser", and "mopedrider." I have condensed these into "pedestrian", "cyclist", and "other" for simplicity. Here is an example of their annotated images:

![Data Exploration](./readme-images/data-exploration.gif)

_Citation: X. Li, F. Flohr, Y. Yang, H. Xiong, M. Braun, S. Pan, K. Li and D. M. Gavrila, "A new benchmark for vision-based cyclist detection", In proceedings of IEEE Intelligent Vehicles Symposium (IV), pages 1028-1033, June 2016_

## Faster R-CNN

![Faster R-CNN Architecture](./readme-images/faster-rcnn-arch.png)

_Credit: Jonathan Hui on Medium ([link](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c))_

Faster R-CNN is a two-stage object detection model that is built on top of Fast R-CNN and R-CNN (or Region-based CNN). Albeit slower than SSD, Faster R-CNN can be more accurate for identifying small objects (especially using FPN). The pretrained model that I've chosen to use is the **fasterrcnn_resnet50_fpn** from PyTorch with 4 components:

1. Image preprocessing
2. ResNet50 backbone with a 4-layer Feature Pyramid Network (FPN)
3. Region Proposal Network (RPN), upon which Regions of Interest (ROIs) are generated based on objectness and anchor-based bounding boxes
4. ROI heads, where ROIs are pooled and fed into FC layers for class and bounding box predictions

To adapt this model for cyclist detection (and pedestrian / other), I fine-tuned the model on 100 new images (with a batch size of 5) for 10 epochs. It's interesting to note here that backpropagation is dependent on 4 loss gradients: 2 for RPN and 2 for ROI. Here is what the training curve looks like after adding all 4 losses together:

<img src="./readme-images/faster-rcnn-curve.png" width=400>

To evaluate the model's performance, we can use a metric called Mean Average Precision (mAP). For simplicity, let's just visualize the classes and bounding boxes that the model outputs. Below is a gif of its predictions on images that were used during training:

![Faster R-CNN Gif 1](./readme-images/faster-rcnn-1.gif)

And its predictions on images that were not used during training:

![Faster R-CNN Gif 2](./readme-images/faster-rcnn-2.gif)

![Faster R-CNN Gif 3](./readme-images/faster-rcnn-3.gif)

While not perfect, the model seems to perform relatively well on both in-sample and out-of-sample data. In particular, its bounding boxes are able to pick up on small objects accurately, but the model has trouble differentiating between incoming cyclists and pedestrians.
