# Light UNet for Satellite Image Segmentation

A Tensorflow implentation of light UNet semantic segmentation framework.

The framework was used in 2017 CCF BDCI remote sensing image semantic segmentation challenge and achieved 0.891 accuracy.



## Configuration Environment

Ubuntu 16.04 + python2.7 + tensorflow1.3 + opencv3.2 + cuda8.0 

This project implement by gpu version of tensorflow1.3. Therefore a Nvidia GPU is needed.

## Installation

1. Clone the repository

   ```shell
   git clone https://github.com/YudeWang/UNet-Satellite-Image-Segmentation.git
   ```

2. Install PyDenseCRF

   You can follow the install instruction of [PyDenseCRF](https://github.com/lucasb-eyer/pydensecrf)

   If you **do not have the permission of sudo**, you can download the source code by:

   ```shell
   git clone https://github.com/lucasb-eyer/pydensecrf.git
   ```

   Follow the instruction and install:

   ```shell
   cd pydensecrf-master
   python setup.py install
   ```

3. Download dataset and model

   You can download 2017 CCF BDCI remote sensing challenge dataset and our pre-trained model from [here](https://drive.google.com/file/d/1FMRMe4qSI-JS6AzrO8kASO3BfHOLoUfM/view). Please unzip package in this repository folder and change the ckpt file name to **UNet_ResNet_itr100000.ckpt**(I used to call it FPN, while the structure of network is symmetrical and then rename it).


## Network Structure

This network use Feature Pyramid Network architecture, each up-sampling layer use linear interpolation instead of de-convolution. Convolution structure we use residual-block, which including convolution and down-sampling (convolution with stride=2). A condition random field(CRF) is added at the end of network with size 256\*256\*512. The loss function is soft-max cross-entropy.

The detail of network architecture can be found in factory.py



## Dataset

The dataset can be found in [here](https://github.com/linsong8208/Satellite-image-segment/tree/master/BDCI/0_data).

Original training data and label is given by png format, each pixel has RGB information. 

In **BDCI-jiage** folder, the labels are plane(1), **road(2), building(3), water(4)**, and the other(0);

In **BDCI-jiage-Semi** folder, the labels are plane(1), **building(2), water(3), road(4)**, and  the other(0).

To generate training dataset, we random select 1024\*1024 patch of original map and scale it into 256\*256. For data augmentation, four kinds of rotation transformation( 0, 90, 180, 270 degree) and minor transformation are applied. You can use following instruction to generate TFRecord format dataset.

```shell
python dataset.py
```



## Train

You can run train.py for training, but **please check training parameters at first**. This code can run on single GPU by following instruction:

```shell
python train.py --gpu=0
```

Training result model will be saved in model folder with name UNet\_ResNet\_itrxxxxxx.ckpt



## Test

We provide pre-trained model **UNet_ResNet_itr100000.ckpt**.

You can use test.py to generate segmentation result.

```shell
python test.py --gpu=0
```

The test result picture can be found in BDCI2017-jiage-Semi/test/x_result.png



<div align="left"> 

<img src="https://github.com/YudeWang/UNet-Satellite-Image-Segmentation/blob/master/sample_visible.png?raw=true" height="40%" width="40%">    <img src="https://github.com/YudeWang/UNet-Satellite-Image-Segmentation/blob/master/sample_result.png?raw=true" height="40%" width="40%">

</div>

## References
1. K. He, X. Zhang, S. Ren, and J. Sun, “[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385),” arXiv:1512.03385, 2015.
2. Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie,"[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)," arXiv:1612.03144,2016. 
3. Olaf Ronneberger, Philipp Fischer, Thomas Brox, "[U-Net: Convolutional Networks for Biomedical Image Segmentation.]( https://arxiv.org/abs/1505.04597)," arXiv:1505.04597.
