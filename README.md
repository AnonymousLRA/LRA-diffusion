# LRA-diffusion
This is the source code of the Label-Retrieval-Augmented Diffusion Models for learning with noisy labels
.
<object data="DDIM_TSNE.pdf" type="DDIM_TSNE.pdf" width="700px" height="700px">
    <embed src="DDIM_TSNE.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="DDIM_TSNE.pdf">Download PDF</a>.</p>
    </embed>
</object>

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

## 3. Generate the Poly-Margin Diminishing (PMD) Noisy Labels
The noisy labels used in our experiments are provided in folder `noisy_labels`.

