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
import glob
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.model_selection import KFold
import os
import argparse

class Dataset(Dataset):
    def __init__(
            self,
            image_folder,
            image_size,
            train_split,
            anno_fold
    ):
        super().__init__()
        self.image_folder = image_folder
        self.image_size = image_size
        self.train_split = train_split
        self.anno_fold = anno_fold
        self.image_paths = []

        for one_split in self.train_split:
            anno_file = os.path.join(self.anno_fold, str(one_split) + '.txt')
            with open(anno_file, 'r') as f:
                for line in f.readlines():
                    image_path = line.split()[0]
                    label = int(line.split()[1])
                    self.image_paths.append([image_path, label])
        print('original number:', len(self.image_paths))

        self.transform = T.Compose([
            T.Resize([image_size, image_size]),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_folder, self.image_paths[index][0])
        image_label = self.image_paths[index][1]
        image = Image.open(image_path)
        return (self.transform(image), image_label)
        
def train(args):
    # Load dataset
    device = torch.device(args.device)
    batch_size = args.batch_size


    image_folder = args.image_fold
    anno_fold = args.anno_fold
    save_fold=args.save_fold
    
    image_size = 224
    epochs = 100
    lr=1e-4
    momentum=0.9
    decay=5e-4


    update_every = 20

    print(device)
    #print(image_folder)
    #print(anno_fold)

    augment_horizontal_flip = True

    

    # Load MNIST dataset
    k_fold=5
    for fold in range(0,k_fold):

        train_split=[]
        test_split=[]
        for j in range(k_fold):
            if j==fold:
                test_split.append(fold)
            else:
                train_split.append(j)
        print('using train fold: ',train_split)
        print('using test fold: ', test_split)

        ds = Dataset(image_folder, image_size, train_split,anno_fold)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

        #print(len(ds))

        # Train network
        class_net = models.densenet201(pretrained=True)
        num_ftrs = class_net.classifier.in_features
        #class_net.classifier = nn.Linear(num_ftrs, 7)
        class_net.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 7)
        )
        class_net.to(device)
        class_net.train()

        #torch.save(class_net.state_dict(), os.path.join(save_fold,'best','classifier_fold_'+str(fold)+'_test.pth'))

        class_opt = torch.optim.SGD(class_net.parameters(), lr=lr, momentum=momentum,weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(class_opt, gamma=1-decay)

        min_loss = 10
        for e in range(epochs):
            print(f'Epoch [{e + 1}/{epochs}]')
            losses = []
            batch_bar = tqdm.tqdm(dl)
            for i, batch in enumerate(batch_bar):
                img, labels = batch
                # print(labels)
                img = img.to(device)
                labels = labels.to(device)
                # print(img.device)
                out = class_net(img)
                # print(labels)
                # print(out)

                # Compute loss and backprop
                loss = F.cross_entropy(out, labels)
                class_opt.zero_grad()
                loss.backward()
                class_opt.step()

                losses.append(loss.item())
                if i % update_every == 0:
                    batch_bar.set_postfix({'Loss': np.mean(losses)})
                    losses = []
            if e > 80 and loss < min_loss:
                min_loss = loss
                torch.save(class_net.state_dict(), os.path.join(save_fold,'best','classifier_fold_'+str(fold)+'_best.pth'))

            batch_bar.set_postfix({'Loss': np.mean(losses)})
            losses = []
            print('lr: ', scheduler.get_last_lr()[0])
            scheduler.step()
        torch.save(class_net.state_dict(), os.path.join(save_fold,'last','classifier_fold_'+str(fold)+'_last.pth'))
        print('Saved classifier model fold: ',fold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', type=str, default='cuda:5', help='device')

    parser.add_argument('--batch_size', type=int, default=96, help='input batch size for training')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay', type=float, default=5e-4, help='exp decay of SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--update_every', type=int, default=20, help='update loss every')
    parser.add_argument('--image_fold', type=str, default='/ISIC2018_Task3_Training_Input_Aug', help='train images ')
    parser.add_argument('--anno_fold', type=str, default='/ISIC_2018_5fold annotations', help='5fold annotation fold')
    parser.add_argument('--save_fold', type=str, default='/save_fold', help='save_fold')

    args = parser.parse_args()
    train(args)