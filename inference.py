import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from unet import UNetModel
from diffusion import GaussianDiffusion
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from torch import nn, einsum
import math
import os
from torchvision import models
import random

if __name__ == '__main__':

    # Load dataset
    device = torch.device('cuda:4')
    image_folder = '/home/ylindq/YUANQIANSHI/DATA/ISIC2018_Task3_Training_Input'
    ckpt='/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/models/isic18_unet-32.pth'
    anno_fold = '/home/ylindq/YUANQIANSHI/DATA/ISIC_2018_5fold'
    cls_ckpt='/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/models/classifier_39.pth'
    print(device)

    # super paparam s
    image_size=256
    augment_horizontal_flip=False
    train_split=[1,2,3,4]
    test_split=[0]
    all_split=[0,1,2,3,4]

    class_net=models.resnet18(pretrained=False)
    fc_inputs = class_net.fc.in_features
    class_net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 7),
        nn.LogSoftmax(dim=1)
    )
    class_net.load_state_dict(torch.load(cls_ckpt, map_location=device))
    class_net.to(device)
    class_net.eval()

    transform = T.Compose([
        T.CenterCrop(450),
        T.Resize(image_size),
        T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    file_list=[]
    for each_split in all_split:
        txt_path=os.path.join(anno_fold,str(each_split)+'.txt')
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                if line.split()[1]!='3':    #ak
                    image_path=line.split()[0]
                    file_list.append(image_path)
    print('all images: ',len(file_list))
    print('this class: ',10015-len(file_list))

    diff_net = UNetModel(image_size=256, in_channels=3, out_channels=3, 
                     model_channels=256, num_res_blocks=2, channel_mult=(1, 1, 2, 2, 4, 4),
                     attention_resolutions=[32,16,8], num_head_channels=64, dropout=0.1, resblock_updown=True, use_scale_shift_norm=True).to(device)
    diff_net.load_state_dict(torch.load(ckpt, map_location=device))
    diff_net.eval()
    diffusion = GaussianDiffusion(T=1000, schedule='linear')

    class InferenceModel(nn.Module):
        def __init__(self,xt):
            super(InferenceModel, self).__init__()
            # Inferred image
            self.img = nn.Parameter(torch.tensor(xt))
            self.img.requires_grad = True
            
        def encode(self):
            return self.img

    #steps = 200  
    back_step=50
    start_t = 200
    forward_t= 300
    steps=start_t
    # start t是finetune步数
    # forward t是加噪步数
    # forward-start=loss部分的理论步数
    # back_step是loss部分的实际步数


    random.shuffle(file_list)
    file_list_bar=tqdm.tqdm(file_list)
    
    for image_name in file_list_bar:
        # read in image
        image_path= os.path.join(image_folder,image_name)
        image=Image.open(image_path)
        image.save('/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/loss_tune/origin.png')
        image=transform(image)
        image = torch.unsqueeze(image, 0).to(device)
        #image=torch.astensor(image, dtype=torch.float)
        image=image.clone().detach().to(device)

        # diffusion q sample
        forward_t = np.array([forward_t],dtype=int)
        xt, epsilon = diffusion.sample(image,forward_t)
        
        # save xt
        image=xt.cpu().numpy()
        image=np.clip(image,-1,1)
        image=np.squeeze(np.uint8((image+1)/2*255)).transpose([1, 2, 0])
        image=Image.fromarray(image)
        image.save('/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/loss_tune/origin_noised.png')

        #initialize differenciable image Model
        xt=xt.cpu().numpy()
        #print(xt.shape)
        model = InferenceModel(xt).to(device)    
        model.train()
        ini_lr=1.2
        opt = torch.optim.Adamax(model.parameters(), lr=ini_lr)
        
        # loss back
        #loss_bar=tqdm.tqdm(range(back_step))
        #for i, _ in enumerate(loss_bar):  
        for i, _ in enumerate(range(back_step)):  
            # Select t      
            t = (back_step - i)/ back_step* (forward_t-start_t)+ start_t   # t从200-100
            t = np.clip(t, 1, diffusion.T)
            t = np.array([t], dtype = int)#.astype(int)
            
            # Denoise by loss back
            sample_img = model.encode()
            #print(sample_img.shape)
            #print(t)
            xt, epsilon = diffusion.sample(sample_img, t)       
            t = torch.from_numpy(t).float().view(1)    
            pred = diff_net(xt.float(), t.to(device))   
            epsilon_pred = pred.float() # Use predicted noise only
            epsilon=epsilon.float()

            
            # Compute diffusion loss
            loss = F.mse_loss(epsilon_pred, epsilon)#.float() 

            # Compute EMA of diffusion loss gradient norm
            opt.zero_grad()

            loss.backward()
            opt.step()
            ''''''

            ''''''
            # class loss
            # 不用class loss的时候好像不用开
            set_label=3
            with torch.no_grad():
                grad_norm = torch.linalg.norm(model.img.grad)
                if i > 0:
                    alpha = 0.5
                    norm_track = alpha*norm_track + (1-alpha)*grad_norm
                else:
                    norm_track = grad_norm

            set_label=torch.tensor([set_label]).to(device)
            sample_img_clipped = torch.clip(model.encode(), -1, 1).to(device)
            cls_out = class_net(sample_img_clipped.float())
            # print(cls_out.dtype)
            pred = F.softmax(cls_out, dim=1)
            pred=pred.squeeze()
            #print(pred)
            #loss=0.01*(1-pred[set_label])

            loss = F.cross_entropy(cls_out.squeeze(-1), set_label)
            opt.zero_grad()
            loss.backward()

            # Clip attribute loss gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3*norm_track)
            opt.step()
            

            lr=ini_lr-i/back_step*ini_lr*0.5
            for param_group in opt.param_groups:
                param_group["lr"] = lr

            '''
            if (i)%10==0:
                with torch.no_grad():
                    image=model.encode()[0].detach().cpu().numpy().transpose([1, 2, 0])
                    image=np.clip(image,-1,1)
                    image=np.squeeze(np.uint8((image+1)/2*255))
                    image=Image.fromarray(image)
                    image.save('/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/loss_tune/'+str(int((back_step - i)/ back_step* (forward_t-start_t)+ start_t))+'.png')
            '''

        # finetune 200 steps
        x = model.encode().float().to(device)
        with torch.no_grad():
            fine_tuned = diffusion.inverse(diff_net, shape=(3,256,256), start_t=start_t, steps=steps, x=x, device=device)
            image=fine_tuned.detach().cpu().numpy()[0].transpose([1, 2, 0])
            image=np.clip(image,-1,1)
            image=np.squeeze(np.uint8((image+1)/2*255))
            image=Image.fromarray(image)
            name=image_name.split('.')[0]+'_3'+'.png'
            #image.save(os.path.join('/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/AK',image_name))
            image.save('/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/AK/'+name)
    