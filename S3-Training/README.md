# S3: Sign-Sparse-Shift Reparametrization for Effective Training of Low-bit Shift Networks

This repository is the DEMO code of the NeurIPS 2021 paper [S3: Sign-Sparse-Shift Reparametrization for Effective Training of Low-bit Shift Networks](https://proceedings.neurips.cc/paper/2021/file/7a1d9028a78f418cb8f01909a348d9b2-Paper.pdf).

Shift neural networks reduce computation complexity by removing expensive multiplication operations and quantizing continuous weights into low-bit discrete values, which are fast and energy-efficient compared to conventional neural networks. However, existing shift networks are sensitive to the weight initialization and yield a degraded performance caused by vanishing gradient and weight sign freezing problem. To address these issues, we propose S3 re-parameterization, a novel technique for training low-bit shift networks. Our method decomposes a discrete parameter in a sign-sparse-shift 3-fold manner. This way, it efficiently learns a low-bit network with weight dynamics similar to full-precision networks and insensitive to weight initialization. Our proposed training method pushes the boundaries of shift neural networks and shows 3-bit shift networks compete with their full-precision counterparts in terms of top-1 accuracy on ImageNet.

<p align="center">
<img src="figures/S3-Shift3bit-Training.png" alt="Training Diagram of S3 re-parameterized 3-bit shift network" width="540">
</p>

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- Download [PyTorch official ImageNet training example](https://github.com/pytorch/examples/tree/master/imagenet).
  - `wget https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py`
  - `wget https://raw.githubusercontent.com/pytorch/examples/master/imagenet/requirements.txt`
- `pip install -r requirements.txt`
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Training from scratch

### 3-bit Shift Network on ResNet-18 ImageNet

To train a 3bit S3 re-parameterized shift network with ResNet18 on ImageNet from scratch, run:
```train
python main.py /path/to/imagenet
```

## Pre-trained checkpoints

Two pre-trained checkpoints corresponding to the 3-bit results reported in the paper are provided [here](./pre-trained-ckpt-evaluation/pre-trained-ckpts.7z). The checkpoints convert into a format compatible with the PyTorch official ImageNet training example so that this open-source code can evaluate the validation accuracy of the checkpoints.

### 3-bit Shift Network on ResNet-18 ImageNet
<details>
To evaluate the pre-trained checkpoint of 3bit S3 re-parameterized shift network with ResNet-18 on ImageNet, run:

```eval
python main_eval.py --evaluate --resume s3-3bit-resnet18-pytorch-imagenet.pth.tar --arch resnet18 /path/to/imagenet
```

Outputs:
```example_output
=> creating model 'resnet18'
=> loading checkpoint 's3-3bit-resnet18-pytorch-imagenet.pth.tar'
=> loaded checkpoint 's3-3bit-resnet18-pytorch-imagenet.pth.tar' (epoch 199)
Test: [  0/196] Time  4.506 ( 4.506)    Loss 6.7311e-01 (6.7311e-01)    Acc@1  82.81 ( 82.81)   Acc@5  96.09 ( 96.09)
Test: [ 10/196] Time  0.072 ( 0.986)    Loss 1.2426e+00 (9.1206e-01)    Acc@1  69.14 ( 77.73)   Acc@5  88.67 ( 92.68)
Test: [ 20/196] Time  2.218 ( 0.904)    Loss 8.9948e-01 (9.1190e-01)    Acc@1  81.64 ( 77.40)   Acc@5  91.02 ( 92.47)
Test: [ 30/196] Time  0.072 ( 0.809)    Loss 8.3274e-01 (8.8309e-01)    Acc@1  80.47 ( 77.95)   Acc@5  94.92 ( 92.97)
Test: [ 40/196] Time  2.417 ( 0.815)    Loss 9.7517e-01 (9.2135e-01)    Acc@1  75.78 ( 76.59)   Acc@5  94.14 ( 93.16)
Test: [ 50/196] Time  0.072 ( 0.775)    Loss 6.4144e-01 (9.1274e-01)    Acc@1  83.59 ( 76.49)   Acc@5  96.88 ( 93.35)
Test: [ 60/196] Time  2.033 ( 0.795)    Loss 1.1415e+00 (9.1855e-01)    Acc@1  70.31 ( 76.14)   Acc@5  91.02 ( 93.45)
Test: [ 70/196] Time  0.072 ( 0.791)    Loss 9.0677e-01 (9.0656e-01)    Acc@1  73.83 ( 76.42)   Acc@5  94.14 ( 93.54)
Test: [ 80/196] Time  0.848 ( 0.784)    Loss 1.7171e+00 (9.2979e-01)    Acc@1  57.03 ( 75.85)   Acc@5  85.55 ( 93.26)
Test: [ 90/196] Time  1.443 ( 0.785)    Loss 2.2276e+00 (9.9383e-01)    Acc@1  51.56 ( 74.53)   Acc@5  75.78 ( 92.41)
Test: [100/196] Time  1.244 ( 0.767)    Loss 1.7705e+00 (1.0593e+00)    Acc@1  55.47 ( 73.16)   Acc@5  82.03 ( 91.58)
Test: [110/196] Time  0.449 ( 0.771)    Loss 1.2247e+00 (1.0864e+00)    Acc@1  70.70 ( 72.59)   Acc@5  89.84 ( 91.17)
Test: [120/196] Time  0.074 ( 0.763)    Loss 1.9402e+00 (1.1115e+00)    Acc@1  55.86 ( 72.25)   Acc@5  76.95 ( 90.76)
Test: [130/196] Time  0.071 ( 0.774)    Loss 1.0368e+00 (1.1486e+00)    Acc@1  74.61 ( 71.40)   Acc@5  92.97 ( 90.29)
Test: [140/196] Time  0.072 ( 0.754)    Loss 1.4686e+00 (1.1709e+00)    Acc@1  65.23 ( 70.97)   Acc@5  83.98 ( 90.00)
Test: [150/196] Time  0.073 ( 0.763)    Loss 1.4905e+00 (1.1954e+00)    Acc@1  69.92 ( 70.51)   Acc@5  85.16 ( 89.62)
Test: [160/196] Time  0.073 ( 0.754)    Loss 1.1636e+00 (1.2138e+00)    Acc@1  73.44 ( 70.19)   Acc@5  89.84 ( 89.35)
Test: [170/196] Time  0.072 ( 0.755)    Loss 7.5062e-01 (1.2348e+00)    Acc@1  77.73 ( 69.75)   Acc@5  96.48 ( 89.05)
Test: [180/196] Time  0.073 ( 0.745)    Loss 1.3958e+00 (1.2521e+00)    Acc@1  64.06 ( 69.37)   Acc@5  88.67 ( 88.84)
Test: [190/196] Time  0.072 ( 0.748)    Loss 1.2849e+00 (1.2503e+00)    Acc@1  64.84 ( 69.33)   Acc@5  92.58 ( 88.89)
 * Acc@1 69.508 Acc@5 88.968
```

The elements of following weight tensors in the checkpoint are restricted to the discrete weight values of 3-bit shift network {-4, -2, -1, 0, 1, 2, 4}
<details>
<summary markdown="span"> Quantized tensor name in ResNet-18 checkpoint </summary>
module.layer1.0.conv1.weight  <br />
module.layer1.0.conv2.weight  <br />
module.layer1.1.conv1.weight  <br />
module.layer1.1.conv2.weight  <br />
module.layer2.0.conv1.weight  <br />
module.layer2.0.conv2.weight  <br />
module.layer2.0.downsample.0.weight  <br />
module.layer2.1.conv1.weight  <br />
module.layer2.1.conv2.weight  <br />
module.layer3.0.conv1.weight  <br />
module.layer3.0.conv2.weight  <br />
module.layer3.0.downsample.0.weight  <br />
module.layer3.1.conv1.weight  <br />
module.layer3.1.conv2.weight  <br />
module.layer4.0.conv1.weight  <br />
module.layer4.0.conv2.weight  <br />
module.layer4.0.downsample.0.weight  <br />
module.layer4.1.conv1.weight  <br />
module.layer4.1.conv2.weight  <br />
</details>

The following code snippet can load a discrete weight tensor from the checkpoint and output the unique discrete values in this tensor.
```eval
import torch
TENSOR_NAME = "module.layer1.0.conv1.weight"
CKPT_NAME = "s3-3bit-resnet18-pytorch-imagenet.pth.tar"

checkpoint = torch.load(CKPT_NAME)
model_state_dict = checkpoint['state_dict']
discrete_weight = model_state_dict[TENSOR_NAME]
print(torch.unique(discrete_weight))
```

Outputs:
```example_output
tensor([-4., -2., -1., -0.,  1.,  2.,  4.], device='cuda:0')
```
</details>

### 3-bit Shift Network on ResNet-50 ImageNet
<details>
To evaluate the pre-trained checkpoint of 3bit S3 re-parameterized shift network with ResNet-50 on ImageNet, run:

```eval
python main_eval.py --evaluate --resume s3-3bit-resnet50-pytorch-imagenet.pth.tar --arch resnet50 /path/to/imagenet
```

Outputs:
```example_output
=> creating model 'resnet50'
=> loading checkpoint 's3-3bit-resnet50-pytorch-imagenet.pth.tar'
=> loaded checkpoint 's3-3bit-resnet50-pytorch-imagenet.pth.tar' (epoch 199)
Test: [  0/196] Time  4.976 ( 4.976)    Loss 4.9636e-01 (4.9636e-01)    Acc@1  86.33 ( 86.33)   Acc@5  97.27 ( 97.27)
Test: [ 10/196] Time  0.221 ( 0.972)    Loss 1.0587e+00 (6.8706e-01)    Acc@1  75.39 ( 82.07)   Acc@5  92.19 ( 95.63)
Test: [ 20/196] Time  1.160 ( 0.907)    Loss 7.0471e-01 (6.8882e-01)    Acc@1  86.33 ( 81.99)   Acc@5  92.58 ( 95.48)
Test: [ 30/196] Time  0.221 ( 0.873)    Loss 8.0941e-01 (6.5377e-01)    Acc@1  78.91 ( 83.09)   Acc@5  94.92 ( 95.60)
Test: [ 40/196] Time  2.344 ( 0.906)    Loss 6.5837e-01 (6.8861e-01)    Acc@1  82.03 ( 81.85)   Acc@5  96.88 ( 95.61)
Test: [ 50/196] Time  0.223 ( 0.829)    Loss 4.6707e-01 (6.8241e-01)    Acc@1  88.67 ( 81.78)   Acc@5  96.88 ( 95.76)
Test: [ 60/196] Time  1.323 ( 0.812)    Loss 8.7407e-01 (6.9512e-01)    Acc@1  74.22 ( 81.40)   Acc@5  96.48 ( 95.87)
Test: [ 70/196] Time  2.609 ( 0.832)    Loss 7.4790e-01 (6.8027e-01)    Acc@1  76.95 ( 81.63)   Acc@5  96.88 ( 96.06)
Test: [ 80/196] Time  0.221 ( 0.810)    Loss 1.4313e+00 (7.0608e-01)    Acc@1  65.23 ( 81.13)   Acc@5  87.11 ( 95.75)
Test: [ 90/196] Time  3.314 ( 0.842)    Loss 1.8285e+00 (7.5399e-01)    Acc@1  58.20 ( 80.08)   Acc@5  85.94 ( 95.25)
Test: [100/196] Time  0.219 ( 0.825)    Loss 1.2244e+00 (8.0642e-01)    Acc@1  66.80 ( 78.93)   Acc@5  89.84 ( 94.59)
Test: [110/196] Time  3.015 ( 0.847)    Loss 8.3800e-01 (8.3314e-01)    Acc@1  78.91 ( 78.41)   Acc@5  94.92 ( 94.27)
Test: [120/196] Time  0.219 ( 0.844)    Loss 1.2821e+00 (8.4899e-01)    Acc@1  71.48 ( 78.15)   Acc@5  88.28 ( 94.02)
Test: [130/196] Time  2.935 ( 0.857)    Loss 6.7108e-01 (8.8153e-01)    Acc@1  81.64 ( 77.40)   Acc@5  95.31 ( 93.68)
Test: [140/196] Time  0.222 ( 0.852)    Loss 1.1377e+00 (8.9882e-01)    Acc@1  72.27 ( 77.09)   Acc@5  91.80 ( 93.49)
Test: [150/196] Time  2.446 ( 0.858)    Loss 1.1069e+00 (9.1730e-01)    Acc@1  76.17 ( 76.76)   Acc@5  90.62 ( 93.22)
Test: [160/196] Time  0.220 ( 0.847)    Loss 7.7915e-01 (9.3251e-01)    Acc@1  83.20 ( 76.46)   Acc@5  93.36 ( 93.00)
Test: [170/196] Time  2.340 ( 0.852)    Loss 5.5731e-01 (9.4940e-01)    Acc@1  84.77 ( 76.01)   Acc@5  96.88 ( 92.81)
Test: [180/196] Time  0.221 ( 0.845)    Loss 1.2214e+00 (9.6362e-01)    Acc@1  67.97 ( 75.69)   Acc@5  93.75 ( 92.70)
Test: [190/196] Time  2.750 ( 0.848)    Loss 1.1438e+00 (9.6272e-01)    Acc@1  69.92 ( 75.63)   Acc@5  94.53 ( 92.75)
 * Acc@1 75.748 Acc@5 92.800
```

The elements of following weight tensors in the checkpoint are restricted to the discrete weight values of 3-bit shift network {-4, -2, -1, 0, 1, 2, 4}
<details>
<summary markdown="span"> Quantized tensor name in ResNet-50 checkpoint </summary>
module.layer1.0.conv1.weight <br />
module.layer1.0.conv2.weight <br />
module.layer1.0.conv3.weight <br />
module.layer1.0.downsample.0.weight <br />
module.layer1.1.conv1.weight <br /> 
module.layer1.1.conv2.weight <br />
module.layer1.1.conv3.weight <br />
module.layer1.2.conv1.weight <br />
module.layer1.2.conv2.weight <br />
module.layer1.2.conv3.weight <br />
module.layer2.0.conv1.weight <br />
module.layer2.0.conv2.weight <br />
module.layer2.0.conv3.weight <br />
module.layer2.0.downsample.0.weight <br />
module.layer2.1.conv1.weight <br />
module.layer2.1.conv2.weight <br />
module.layer2.1.conv3.weight <br />
module.layer2.2.conv1.weight <br />
module.layer2.2.conv2.weight <br />
module.layer2.2.conv3.weight <br />
module.layer2.3.conv1.weight <br />
module.layer2.3.conv2.weight <br />
module.layer2.3.conv3.weight <br />
module.layer3.0.conv1.weight <br />
module.layer3.0.conv2.weight <br />
module.layer3.0.conv3.weight <br />
module.layer3.0.downsample.0.weight <br />
module.layer3.1.conv1.weight <br />
module.layer3.1.conv2.weight <br />
module.layer3.1.conv3.weight <br />
module.layer3.2.conv1.weight <br />
module.layer3.2.conv2.weight <br />
module.layer3.2.conv3.weight <br />
module.layer3.3.conv1.weight <br />
module.layer3.3.conv2.weight <br />
module.layer3.3.conv3.weight <br />
module.layer3.4.conv1.weight <br />
module.layer3.4.conv2.weight <br />
module.layer3.4.conv3.weight <br />
module.layer3.5.conv1.weight <br />
module.layer3.5.conv2.weight <br />
module.layer3.5.conv3.weight <br />
module.layer4.0.conv1.weight <br />
module.layer4.0.conv2.weight <br />
module.layer4.0.conv3.weight <br />
module.layer4.0.downsample.0.weight <br />
module.layer4.1.conv1.weight <br />
module.layer4.1.conv2.weight <br />
module.layer4.1.conv3.weight <br />
module.layer4.2.conv1.weight <br />
module.layer4.2.conv2.weight <br />
module.layer4.2.conv3.weight <br />
</details>
</details>

## Results

Our model achieves the following performance on :

### Image Classification on ImageNet

#### Results in the paper
<p align="left">
<img src="figures/tables2.png" alt="Table-1-2" width="450">
</p>

#### Evaluation code output
| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| 3-bit Shift ResNet-18 |     69.508%         |      88.968%       |
| 3-bit Shift ResNet-50 |     75.748%         |      92.800%       |

The minor accuracy difference (~0.3%) between Table 1 and the evaluation code output may cause by the difference between our implementation and the PyTorch official ImageNet training example.