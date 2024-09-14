# Multilevel Support-Assisted Prototype Optimization Network for Few-Shot Lesion Segmentation in Lung CT Images<br>
This repo contains code for our paper Multilevel Support-Assisted Prototype Optimization for Few-Shot Lesion Segmentation in Lung CT Images<br>

## Abstract<br>
Medical image annotation for lesion segmentation is scarce and costly, particularly for lung diseases. This paper presents MSPO-Net, a novel multilevel support-assisted prototype optimization network for few-shot lesion segmentation in lung CT images. MSPO-Net leverages multilevel prototypes learned from low-level to high-level features, coupled with a self-attention strategy to adaptively focus on global information. An optimization module further refines the prototypes using support annotations, enhancing their representativeness. Experiments on a small-scale CT dataset annotated by experienced doctors demonstrate that MSPO-Net achieves state-of-the-art performance in segmenting lesions from single and unseen lung diseases, promising to reduce doctors' workload and improve diagnostic accuracy. Our code and dataset are openly accessible to facilitate future research.<br>



## Dependencies<br>
* Python==3.8<br>
* PyTorch==1.0.1<br>
* torchvision==0.2.1<br>
* NumPy==1.22.0<br>
* sacred==0.7.5<br>
* tqdm==4.32.2<br>
* SciPy, PIL<br>

## Data<br>
Download CT images of secondary pulmonary tuberculosis, pulmonary aspergillosis, lung adenocarcinoma cases from here<br>
[Lung-Diseases-CT](URL "https://github.com/Tian-Yuan-ty/Lung-Diseases-CT")

## Usage<br>
Download the ImageNet-pretrained weights of VGG16 network from torchvision: <br>
[https://download.pytorch.org/models/vgg16-397923af.pth](URL "https://download.pytorch.org/models/vgg16-397923af.pth") and put it under `MSPO-Net/pretrained_model` folder.

Change configuration via config.py, then train the model using python train.py or test the model using python test.py.<br>
