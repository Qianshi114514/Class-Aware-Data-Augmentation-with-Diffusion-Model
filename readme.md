# Class Aware Data Augmentation with Diffusion Model


This is the official code for "Class Aware Data Augmentation with Diffusion Model".

# Pipeline
![pipeline](figs/CADA.png)


# Requirements
matplotlib==3.4.3
numpy==1.20.3
Pillow==9.5.0
scikit_learn==1.2.2
torch==1.13.1
torchvision==0.9.0+cu111
tqdm==4.62.3

details in requirements.txt

# Usage
## generate augmentation images

1.download isic-2018 dataset and unzip, remove the first row of the annotation csv

2.use split_isic_2018.py to split the annotation into 5 txt as the 5-fold annotations

3.use train_diffusion_isic.py and train_guidance_classifier.py to train the unconditional ddpm and guiding classifier (here we adopt resnet-18 for easy training)

4.use inference.py to generate augmentation images, after that please drag all the images into the fold of "ISIC2018_Task3_Training" (and you can further rename it as "ISIC2018_Task3_Training_Aug")

## compare classifier performance on different aumentation methods

5.

# Citation
[ ] TO BE RELEASED.

# TODO
- [ ] Refactor the code to make it more readable.

# Acknowledgment 
