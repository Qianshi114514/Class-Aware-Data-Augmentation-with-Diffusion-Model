# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
import math
from PIL import Image
from unet import UNetModel
from diffusion import GaussianDiffusion
import glob
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.model_selection import KFold
import os
import argparse
import warnings

class Dataset(Dataset):
    def __init__(
            self,
            image_folder,
            image_size,
            train_split,
            anno_fold,
    ):
        super().__init__()
        self.image_folder = image_folder
        self.image_size = image_size
        self.train_split=train_split
        self.anno_fold=anno_fold
        self.image_paths=[]

        for one_split in self.train_split:
            anno_file=os.path.join(self.anno_fold,str(one_split)+'.txt')
            with open(anno_file, 'r') as f:
                for line in f.readlines():
                    image_path=line.split()[0]
                    label=int(line.split()[1])
                    self.image_paths.append([image_path,label])

        self.transform = T.Compose([
            T.Resize([image_size,image_size]),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder,self.image_paths[index][0])
        image_label=self.image_paths[index][1]
        image = Image.open(image_path)
        return (self.transform(image),image_label)

def train(args):
    # Load dataset
    device = torch.device(args.device)
    image_folder = args.image_fold
    anno_fold = args.anno_fold
    save_fold=args.save_fold

    batch_size = 128
    image_size = 256
    epochs = 100
    lr=1e-4
    momentum=0.9
    decay=5e-4
    update_every = 20

    print(device)

    train_split=[1,2,3,4]
    test_split=[0]

    print('using train fold: ',train_split)
    print('using test fold: ', test_split)

    ds = Dataset(image_folder, image_size, train_split,anno_fold)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    print('length of this split: ',len(ds))

    # Train network
    class_net = models.resnet18(pretrained=True)
    fc_inputs = class_net.fc.in_features
    class_net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 7),
        nn.LogSoftmax(dim=1)
    )
    class_net.to(device)
    class_net.train()

    #class_opt = torch.optim.Adam(class_net.parameters(), lr=1e-4,)
    class_opt = torch.optim.SGD(class_net.parameters(), lr=lr, momentum=momentum,weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(class_opt, gamma=1-decay)
    min_loss=10

    for e in range(epochs):
        print(f'Epoch [{e + 1}/{epochs}]')
        losses = []
        batch_bar = tqdm.tqdm(dl)
        for i, batch in enumerate(batch_bar):
            img, labels = batch

            img=img.to(device)
            labels =labels.to(device)
            out = class_net(img)

            # Compute loss and backprop
            loss = F.cross_entropy(out, labels)
            class_opt.zero_grad()
            loss.backward()
            class_opt.step()


            losses.append(loss.item())
            if i % update_every == 0:
                batch_bar.set_postfix({'Loss': np.mean(losses)})
                losses = []

        batch_bar.set_postfix({'Loss': np.mean(losses)})
        losses = []
        print('lr: ', scheduler.get_last_lr()[0])
        scheduler.step()
    torch.save(class_net.state_dict(), os.path.join(save_fold,'classifier_'+str(e)+'.pth'))
    print('Saved classifier model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--image_size', type=int, default=256, help='image size')

    parser.add_argument('--image_fold', type=str, default='path/to/ISIC2018_Task3_Training_Input', help='train images')
    parser.add_argument('--anno_fold', type=str, default='path/to/annotation/files', help='5fold annotation fold')
    parser.add_argument('--save_fold', type=str, default='path/to/save/guidance/classifier/models', help='save_fold')

    args = parser.parse_args()
    train(args)