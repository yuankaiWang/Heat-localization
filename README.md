
## Prerequisites

- Python 3.7
- PyTorch 1.4
- Computing device with GPU


## Getting started
### Installation

- (Optional) Install [Anaconda3](https://www.anaconda.com/download/) for managing Python and packages
- Install [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- Install [PyTorch](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 1.4](https://pytorch.org/get-started/previous-versions/)

### Data 
#### Availability
This model was trained on the [National Lung Screening Trial (NLST)](https://biometry.nci.nih.gov/cdas/learn/nlst/images/) dataset. The NLST is made publicly available by the National Cancer Institute.

#### Preprocess
- **Heart Detection**: [RetinaNet](https://github.com/yhenon/pytorch-retinanet) was used in our study for heart detection.
- **Resize & Normalization**: The detected heart region was resized into 128x128x128. The image was normalized with a range of -300HU~500HU.

### Get Trained Model

**BEFORE RUNNING THE CODE, PLEASE DOWNLOAD THE NETWORK CHECKPOINT FIRST.**

The trained model can be downloaded through [this link](https://1drv.ms/u/s!AurT2TsSKdxQvz1aHvmxTlkDNkTz?e=8rCnJl). Please download the checkpoint to the `./checkpoint` folder.


### CVD Risk Prediction

To predict CVD Risk from an image, run:
```bash
python pred.py
```
- `--path` path of the input image. #Default: `./demos/Positive_CAC_1.npy`
- `--iter` iteration of the checkpoint to load. #Default: 8000

#### Input

The model takes a normalized 128x128x128 `numpy.ndarray` as an input, i.e., each item in the `ndarray` ranges 0~1.

#### Output

A real number in \[0, 1\] indicates the estimated CVD risk.

#### Demo

We uploaded 4 demos in the `./demo` folder, including one CVD negative case and three CVD positive case. One of the CVD positive subjects died because of CVD in the trial. 

The name of the file indicates its label and the CAC grade evaluated by our radiologists.
