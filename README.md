# AFFGA-Net

The AFFGA-Net is a high performance network which predicts the quality and pose of grasps at every pixel in an input RGB image. 

This repository contains the data set used to train AFFGA-Net and the program for labeling the grasp model.



**High-performance Pixel-level Grasp Detection based on Adaptive Grasping and Grasp-aware Network**

Dexin Wang, Chunsheng Liu, Faliang Chang, and Nanjun Li

This paper has been accepted by *IEEE Trans. Ind. Electron*.

[TechRxiv](https://www.techrxiv.org/articles/preprint/High-performance_Pixel-level_Grasp_Detection_based_on_Adaptive_Grasping_and_Grasp-aware_Network/14680455) | [Video](https://youtu.be/ccA1jkkbBJA)



## Installation

This code was developed with Python 3.6 on Ubuntu 16.04. The main Python requirements:

```
pytorch==1.2 or higher version
opencv-python
mmcv
numpy
```

## Datasets

1. Download and extract [Cornell and Clutter Dataset](https://github.com/liuchunsense/Clutter-Grasp-Dataset).

2. run `generate_grasp_mat.py`，convert `pcd*Label.txt` to `pcd*grasp.mat`, they represent the same label, but the format is different, which is convenient for AFFGA-Net to read.

3. Put all the samples of the Cornell and Clutter datasets in the same folder, and put train-test folder in the upper directory of the dataset, as follows

   ```
   D:\path_to_dataset\
   ├─cornell_clutter
   │  ├─pcd0100grasp.mat
   │  └─pcd0100r.png
   │  |
   |  └─pcd2000grasp.mat
   |  └─pcd2000r.png
   |
   ├─train-test
   │  ├─train-test-all
   │  ├─train-test-cornell
   │  └─train-test-mutil
   │  └─train-test-single
   |
   ├─other_files
   ```



## Pre-trained Model

Some example pre-trained models for AFFGA-Net can be downloaded from [here](https://github.com/liuchunsense/affga_net/releases/tag/v1.0).

The model is trained on the Cornell and Clutter dataset using the RGB images.

The zip file contains the full saved model from `torch.save(model, path)`.



## Training

Training is done by the `train_net.py` script.

Some basic examples:

```shell
python train_net.py --dataset-path <Path To Dataset>
```

Trained models are saved in `output/models` by default, with the validation score appended.



## Visualisation

visualisation of the trained networks are done using the `demo.py` script. 

Modify the path of the pre-trained model before running, i.e. `model`.

Some output examples of AFFGA-Net is under the `demo\output`.

## Running on a kinova Robot
future work
