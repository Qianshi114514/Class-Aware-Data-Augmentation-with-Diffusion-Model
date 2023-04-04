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
#from eval_utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score,roc_auc_score

import pandas as pd


if __name__ == '__main__':
    device = torch.device('cuda:7')
    image_folder = '/ISIC2018_Task3_Training_Aug'
    anno_fold = '/ISIC_2018_5fold2'
    result_root_fold='/eval_results'
    model_root_fold='/models/best'
    aug_type=['origin','aug_10','aug_50','aug_100','flip','crop','rot','shear','cutout']

    batch_size = 32
    image_size = 256

    print(device)

    augment_horizontal_flip = True


    class Dataset(Dataset):
        def __init__(
                self,
                image_folder,
                image_size,
                train_split,
                anno_fold,
                augment_horizontal_flip=True
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
                # 先裁切再缩放
                # T.CenterCrop(450),
                T.Resize([image_size, image_size]),
                # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
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

    for name in aug_type:
        model_fold=os.path.join(model_root_fold,name)
        #print(model_fold)
        #/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic_dense121/isic_dense121_models/sec_5fold/best/cutout
        model_list=glob.glob(os.path.join(model_fold,'*.pth'))
        #print(model_list)
        result_fold=os.path.join(result_root_fold,name)
        if not os.path.exists(result_fold):
            os.makedirs(result_fold)
        print('name: ',name)
        print('total model number: ',len(model_list))

        total_acc=0
        total_pre=0
        total_kappa=0
        total_f1=0

        cf_mtx=np.zeros((7,7))

        for model in model_list:
            train_split=[]
            test_split=[]
            model_name=model.split('/')[-1]
            #print(model)
            test_split.append(int(model_name[16]))
            #print('using train fold: ',train_split)
            print('using test fold: ', test_split)
            #print('using model: ',model)

            ds = Dataset(image_folder, image_size, test_split,anno_fold,augment_horizontal_flip=augment_horizontal_flip)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)


            # Train network
            class_net = models.resnet50(pretrained=True)
            fc_inputs = class_net.fc.in_features
            class_net.fc = nn.Sequential(
                nn.Linear(fc_inputs, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 7)
            )
            class_net.load_state_dict(torch.load(model, map_location=device))
            class_net.to(device)
            class_net.eval()

            #print('model loaded')

            losses = []
            batch_bar = tqdm.tqdm(dl)
            gt_label=[]
            pred_label=[]

            for i, batch in enumerate(batch_bar):
                img, labels = batch
                # print(labels)
                img = img.to(device)
                #labels = labels.to(device)
                gt_label.extend(labels.numpy().tolist())
                # print(img.device)
                out = class_net(img)
                # print(labels)
                # print(out)
                softMax=nn.Softmax()
                out_score=F.softmax(out.detach().cpu(),dim=1)
                #print(out_score)
                output_index = torch.argmax(out_score, dim=1)
                pred_label.extend(output_index.numpy().tolist())


            acc = accuracy_score(gt_label, pred_label)
            kappa=cohen_kappa_score(gt_label, pred_label)
            #print(acc)
            cm=confusion_matrix(gt_label, pred_label)
            cf_mtx=cf_mtx+cm
            #print(cm)
            #p3 = precision_score(gt_label, pred_label, average='weighted')
            #print(p3)

            #p1 = precision_score(gt_label, pred_label, average='micro')
            #p2 = precision_score(gt_label, pred_label, average='macro')
            p3 = precision_score(gt_label, pred_label, average='weighted')

            #r1 = recall_score(gt_label, pred_label, average='micro')
            #r2 = recall_score(gt_label, pred_label, average='macro')
            #r3 = recall_score(gt_label, pred_label, average='weighted')

            #f1score1 = f1_score(gt_label, pred_label, average='micro')
            #f1score2 = f1_score(gt_label, pred_label, average='macro')
            f1score3 = f1_score(gt_label, pred_label, average='weighted')
            ''''''
            #result_txt=os.path.join(result_fold,model_name.split('.')[0]+'.txt')
            #result_csv=os.path.join(result_fold,model_name.split('.')[0]+'.csv')

            result_txt=os.path.join(result_fold,model_name[:-4]+'.txt')
            result_csv=os.path.join(result_fold,model_name[:-4]+'.csv')
            #print(result_txt)

            #con_mtx=cm.tolist()
            #con_mtx.to_csv(result_csv,encoding='utf-8')
            con_mtx=pd.DataFrame(data=cm,columns=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'],index=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'])
            con_mtx.to_csv(result_csv,encoding='utf-8')
            total_acc=total_acc+acc
            total_pre=total_pre+p3
            total_f1=total_f1+f1score3
            total_kappa=total_kappa+kappa
            ''''''
            with open(result_txt,'a') as f:
                f.write('accuracy: '+str(acc)+'\n')
                f.write('weighted precision: ' + str(p3) + '\n')
                f.write('\n')

                #f.write('micro recall: ' + str(r1) + '\n')
                #f.write('macro recall: ' + str(r2) + '\n')
                #f.write('weighted recall: ' + str(r3) + '\n')
                #f.write('\n')
                f.write('weighted F1: ' + str(f1score3) + '\n')
                f.write('\n')
                f.write('kappa: ' + str(kappa) + '\n')
                f.write('\n')
                f.write('Confusion Matrix:\n ' + str(cm) + '\n')
            #break
            
        total_acc=total_acc/5
        total_pre=total_pre/5
        total_f1=total_f1/5
        total_kappa=total_kappa/5
        result_txt=os.path.join(result_fold,name+'.txt')
        #print(cf_mtx/5)

        result_csv=os.path.join(result_fold,name+'.csv')
        cf_mtx=cf_mtx/5
        con_mtx=pd.DataFrame(data=cf_mtx,columns=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'],index=['MEL','NV','BCC','AKIEC','BKL','DF','VASC'])
        con_mtx.to_csv(result_csv,encoding='utf-8')
        with open(result_txt,'a') as f:
            f.write('accuracy: '+str(total_acc)+'\n')
            f.write('\n')
            f.write('weighted precision: ' + str(total_pre) + '\n')
            f.write('\n')
            f.write('weighted F1: ' + str(total_f1) + '\n')
            f.write('\n')
            f.write('kappa: ' + str(total_kappa) + '\n')
            #f.write('Confusion Matrix:\n ' + str(cm) + '\n')
        