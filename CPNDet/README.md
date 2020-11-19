## Attention ！We failed to open source the trained model! You may need to train by yourselves.

# [Corner Proposal Network for Anchor-free, Two-stage Object Detection](https://arxiv.org/abs/2007.13816)

by [Kaiwen Duan](https://scholar.google.com/citations?hl=zh-CN&user=TFHRaZUAAAAJ&scilu=&scisig=AMD79ooAAAAAXLv9_7ddy26i4c6z5n9agk05m97faUdN&gmla=AJsN-F78W-h98Pb2H78j6lTKbjdn0fklhe2X_8CCPqRU2fC4KJEIbllhD2c5F0irMR3zDiehKt_SH26N2MHI1HlUMw6qRba9HMbiP3vnQfJqD82FrMAPdlU&sciund=10706678259143520926&gmla=AJsN-F5cOpNUdnI6YrZ9joRa6JE2nP6wFKU1GKVkNIfCmmgjk431Lg2BYCS6wn5WWZxdnzBjLfaUwdUJtvPXo53vfoOQoTGP5fHh2X0cCssVtXm8BI4PaM3_oQvKYtCx7o1wivIt1l49sDK6AZPvHLMxxPbC4GbZ1Q&sciund=10445692451499027349), [Lingxi Xie](http://lingxixie.com/Home.html), [Honggang Qi](http://people.ucas.ac.cn/~hgqi), [Song Bai](http://songbai.site/), [Qingming Huang](https://scholar.google.com/citations?user=J1vMnRgAAAAJ&hl=zh-CN) and [Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=zh-CN)

**The code to train and evaluate the proposed CPN is available here. For more technical details, please refer to our [arXiv paper](https://arxiv.org/abs/2007.13816).**

**We thank [Princeton Vision & Learning Lab](https://github.com/princeton-vl) for providing the original implementation of [CornerNet](https://github.com/princeton-vl/CornerNet). We also refer to some codes from [mmdetection](https://github.com/open-mmlab/mmdetection) and [Objects as Points](https://github.com/xingyizhou/CenterNet), we thank them for providing their implementations.**

**CPN is an anchor-free, two-stage detector which gets trained from scratch. On the MS-COCO dataset, CPN achieves an AP of 49.2%, which is competitive among state-of-the-art object detection methods. In the scenarios that require faster inference speed, CPN can be further accelerated by properly replacing with a lighter backbone (e.g., DLA-34) and not using flip augmentation at the inference stage. In this configuration, CPN reports a 41.6 AP at 26.2 FPS (full test) and a 39.7 AP at 43.3 FPS.**

## Abstract

  The goal of object detection is to determine the class and location of objects in an image. This paper proposes a novel anchor-free, two-stage framework which first extracts a number of object proposals by finding potential corner keypoint combinations and then assigns a class label to each proposal by a standalone classification stage. We demonstrate
that these two stages are effective solutions for improving recall and precision, respectively, and they can be integrated into an end-to-end network. Our approach, dubbed Corner Proposal Network (CPN) enjoys the ability to detect objects of various scales and also avoids being confused by a large number of false-positive proposals. On the MS-COCO dataset, CPN achieves an AP of 49.2% which is competitive among state-of-the-art object detection methods. CPN can also fit scenarios that desire for network efficiency. Equipping with a lighter backbone and switching off image flip in inference, CPN achieves 41.6% at 26.2 FPS or 39.7% at 43.3 FPS, surpassing most competitors with the same inference speed.

## Introduction

CPN is a framework for object detection with deep convolutional neural networks. You can use the code to train and evaluate a network for object detection on the MS-COCO dataset.

- It achieves state-of-the-art performance (an AP of 49.2%) on one of the most challenging dataset: MS-COCO.
- It achieves a good trade-off between accuracy and speed (41.6AP/26.2FPS or 39.7AP/43.3FPS).
- At the training stage, we use 8 NVIDIA Tesla-V100 (32GB) GPUs on [HUAWEI CLOUD](https://www.huaweicloud.com/intl/zh-cn/) to train the network, the traing time is about 9 days, 5 days and 3 days for HG104, HG52 and DLA34, respectively.
- Our code is written in Pytorch (the master branch works with **PyTorch 1.1.0**), based on [CornerNet](https://github.com/princeton-vl/CornerNet), [mmdetection](https://github.com/open-mmlab/mmdetection), [Objects as Points](https://github.com/xingyizhou/CenterNet) and [CenterNet](https://github.com/Duankaiwen/CenterNet).

**If you encounter any problems in using our code, please contact Kaiwen Duan: kaiwenduan@outlook.com.**

## Architecture

![Network_Structure](https://xxx/Network_Structure.jpg)

## AP(%) on COCO test-dev and Models  

|                         Backbone                          |                          Input Size                          |  AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| :-------------------------------------------------------: | :----------------------------------------------------------: | :--: | :-------------: | :-------------: | :------------: | :------------: | :------------: |
|                           DLA34                           |                             ori.                             | 41.7 |      58.9       |      44.9       |      20.2      |      44.1      |      56.4      |
| DLA34  ![](http://latex.codecogs.com/gif.latex?\\ddagger) | <a href="https://www.codecogs.com/eqnedit.php?latex=\leq&space;1.8\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\leq&space;1.8\times" title="\leq 1.8\times" /></a> | 44.5 |      62.3       |      48.3       |      25.2      |      46.7      |      58.2      |                                                           |
|                           HG52                            |                             ori.                             | 43.9 |      61.6       |      47.5       |      23.9      |      46.3      |      57.1      |
| HG52  ![](http://latex.codecogs.com/gif.latex?\\ddagger)  | <a href="https://www.codecogs.com/eqnedit.php?latex=\leq&space;1.8\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\leq&space;1.8\times" title="\leq 1.8\times" /></a> | 45.8 |      63.9       |      49.7       |      26.8      |      48.4      |      59.4      |                                                           |
|                           HG104                           |                             ori.                             | 47.0 |      65.0       |      51.0       |      26.5      |      50.2      |      60.7      |
| HG104  ![](http://latex.codecogs.com/gif.latex?\\ddagger) | <a href="https://www.codecogs.com/eqnedit.php?latex=\leq&space;1.8\times" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\leq&space;1.8\times" title="\leq 1.8\times" /></a> | 49.2 |      67.3       |      53.7       |      31.0      |      51.9      |      62.4      |                                                           |

**Notes:**

- ![](http://latex.codecogs.com/gif.latex?\\ddagger) denotse multi-scale testing.

## Comparison of AR(%) on COCO validation set 

|    Method    |  Backbone   |  AR  | AR<sub>1+</sub> | AR<sub>2+</sub> | AR<sub>3+</sub> | AR<sub>4+</sub> | AR<sub>5:1</sub> | AR<sub>6:1</sub> | AR<sub>7:1</sub> | AR<sub>8:1</sub> |
| :----------: | :---------: | :--: | :-------------: | :-------------: | :-------------: | :-------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| Faster R-CNN | X-101-64x4d | 57.6 |      73.8       |      77.5       |      79.2       |      86.2       |       43.8       |       43.0       |       34.3       |       23.2       |
|     FCOS     | X-101-64x4d | 64.9 |      82.3       |      87.9       |      89.8       |      95.0       |       45.5       |       40.8       |       34.1       |       23.4       |
|  CornerNet   |   HG-104    | 66.8 |      85.8       |      92.6       |      95.5       |      98.5       |       50.1       |       48.3       |       40.4       |       36.5       |
|  CenterNet   |   HG-104    | 66.8 |      87.1       |      93.2       |      95.2       |      96.9       |       50.7       |       45.6       |       40.1       |       32.3       |
|     CPN      |   HG-104    | 68.8 |      88.2       |      93.7       |      95.8       |      99.1       |       54.4       |       50.6       |       46.2       |       35.4       |

**Notes:**

- Here, the average recall is recorded for targets of different aspect ratios and different sizes. To explore the limit of the average recall for each method, we exclude the impacts of bounding-box categories and sorts on recall, and compute it by allowing at most 1000 object proposals. AR<sub>1+</sub>, AR<sub>2+</sub>, AR<sub>3+</sub> and AR<sub>4+</sub> denote box area in the ranges of (96<sup>2</sup>, 200<sup>2</sup>], (200<sup>2</sup>, 300<sup>2</sup>], (300<sup>2</sup>, 400<sup>2</sup>] and (400<sup>2</sup>, <a href="https://www.codecogs.com/eqnedit.php?latex=\infty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\infty" title="\infty" /></a>), respectively. 'X' and 'HG' stand for ResNeXt and Hourglass, respectively.

## Inference speed COCO validation set 

| Backbone | Input Size | Flip |  AP  | FPS  |
| :------: | :--------: | :--: | :--: | :--: |
|   HG52   |    ori.    | Yes  | 43.8 | 9.9  |
|   HG52   | 0.7x ori.  |  No  | 37.7 | 24.0 |
|  HG104   |    ori.    | Yes  | 46.8 | 7.3  |
|  HG104   | 0.7x ori.  |  No  | 40.5 | 17.9 |
|  DLA34   |    ori.    | Yes  | 41.6 | 26.2 |
|  DLA34   |    ori.    |  No  | 39.7 | 43.3 |

**Notes:**

- The FPS is measured on an NVIDIA Tesla-V100 GPU on [HUAWEI CLOUD](https://www.huaweicloud.com/intl/zh-cn/).

## Preparation

Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.

```
conda create --name CPN --file conda_packagelist.txt
```

After you create the environment, activate it.

```
source activate CPN
```

## Installing some APIs

```
python setup.py
```

## Downloading MS COCO Data

- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<path>/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<path>/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation

To train CPN104 or CPN52 or CPN_DLA34:

```
python train.py --cfg_file HG104
```

or 

```
python train.py --cfg_file HG52
```

or 

```
python train.py --cfg_file DLA34
```

We provide the configuration file `config/HG104.json`,  `config/HG52.json` and `config/DLA34.json` in this repo. If you want to train you own CPN, please adjust the batch size in corresponding onfiguration files to accommodate the number of GPUs that are available to you. Note that if you want train DLA34, you need to firstly download the [pre-trained model](http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth), and put it under `CPN/cache/nnet/DLA34/pretrain`.

To use the trained model:

```
cd code
```

```
python test.py --cfg_file HG104 --testiter 220000 --split <split>
```

or 

```
python test.py --cfg_file HG52 --testiter 270000 --split <split>
```

or 

```
python test.py --cfg_file DLA34 --testiter 270000 --split <split>

```

where `<split> = validation or testing`.

You need to download the corresponding models and put them under `CPN/cache/nnet`.

You can add `--no_flip` for testing without flip augmentation.

You can also add `--debug` to visualize some detection results (uncomment the codes from line 137 in `CPN/code/test/coco.py`).

We also include a configuration file for multi-scale evaluation, which is `HG104-multi_scale.json` and `HG52-multi_scale.json` and `DLA34-multi_scale.json` in this repo, respectively. 

To use the multi-scale configuration file:

```
python test.py --cfg_file HG104 --testiter <iter> --split <split> --suffix multi_scale

```

or

```
python test.py --cfg_file HG52 --testiter <iter> --split <split> --suffix multi_scale

```

or 

```
python test.py --cfg_file DLA34 --testiter <iter> --split <split> --suffix multi_scale

```



