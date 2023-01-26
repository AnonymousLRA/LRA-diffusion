# LRA-diffusion
This is the source code of the Label-Retrieval-Augmented Diffusion Models for learning with noisy labels.

![CIFAR-10_TSNE.pdf](https://github.com/AnonymousLRA/LRA-diffusion/files/10513073/CIFAR-10_TSNE.pdf)


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
The pre-trianed SimCLR encoder for CIFAR-10 and CIFAR-100 is available at: [SimCLR models](https://drive.google.com/drive/folders/1SXzlQoOAksw349J2jnBSh5aCprDWdTQb?usp=sharing) <br />
Please download the SimCLR models and put them in to the model folder.<br />

CLIP models are available in the python package. Do not need to download manually.

## 3. Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in our experiments are provided in folder `noisy_label`.<br />
The label noise is generated by [PLC](https://github.com/AnonymousLRA/PLC/tree/master/cifar)

## Reference
(PLC) Progressive Label Correction:
```
@article{zhang2021learning,
  title={Learning with feature-dependent label noise: A progressive approach},
  author={Zhang, Yikai and Zheng, Songzhu and Wu, Pengxiang and Goswami, Mayank and Chen, Chao},
  journal={arXiv preprint arXiv:2103.07756},
  year={2021}
}
```
