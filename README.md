# LRA-diffusion
This is the source code of the Label-Retrieval-Augmented Diffusion Models for learning with noisy labels.

![](https://github.com/AnonymousLRA/LRA-diffusion/files/10512654/DDIM_TSNE.pdf)

## 1. preparing python environment
create a virtual environment.<br />
Install and create a virtual environment for python3
```
sudo pip3 install virtualenv
python3 -m venv venv3
```
Activate the virtual environment and install requirements.<br />
```
source ./venv3/bin/activate
pip install -r requirements.txt
```

## 2. Pre-trained model
The pre-trianed SimCLR encoder for CIFAR-10 and CIFAR-100 is available at: <br />
Please download the model and put them in to the model folder.

## 3. Noisy label

