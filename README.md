# ZeroCostDL4Mic_VirtualMultiplexing

## Overview

This repository contains the code as well as the tool generated for the project 'User-friendly deep learning-based morphometric unmixing of multiplex 3D imaging data'. 

It contains: 

* ZeroCostDL4Mic-VirtualMultiplexing. 
* Code to generate training and testing data from CZI format image dataset. 
* Deep Learning training and test examples. 

ZeroCostDL4Mic-VirtualMultiplexing is a tool for generate and use deep learning models for signal unmixing in fluorescent microscopy imaging. It is set in a user friendly interface so anyone can use it, for that it is implemented in [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). 

In the tool, for generating data from a CZI file is used the [DataGenerator.py]([https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/DataGenerator.py](https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Tools/DataGenerator.py)), and the Deep Learning approach used for virtual multiplexing is based on [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 

<img src="https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Images/Data_workflow.jpg" width="650" height="400">

## How to use ZeroCostDL4Mic notebook?

ZeroCostDL4Mic-VirtualMultiplexing notebook can be directly opened from GitHub into Colab by clicking on the link. It is mandatory for use it to create a local copy to your Google Drive. Once you have a copy of it follow the intructions present on the notebook. 
It is necesary to use this tool to have a Google Drive account for having a copy of the notebook as well as for uploading the files that you are going to use. 

### User general workflow

Once you have copied the notebook in your Google Drive account, you can follow the general pipeline implemented shown below:

<img src ="https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Images/User_workflow.jpg" width="650" height="400">

### What steps do you need to follow in the notebook?

If you don't know which steps to follow in the noebook, answer the questions present on the decission tree below to see which steps you need to follow. 

<img src ="https://github.com/akabago/ZeroCostDL4Mic-VirtualMultiplexing/blob/main/Images/User_steps.jpg" width="<550" height="650">


