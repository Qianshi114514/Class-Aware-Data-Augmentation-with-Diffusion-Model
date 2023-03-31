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


mpl.rc('image', cmap='gray')

if __name__ == '__main__':

    # Load dataset
    device = torch.device('cuda:0')
    batch_size = 128
    image_folder = 'path/to/train/images'
    print(device)
    print(image_folder)

    image_size=256
    augment_horizontal_flip=True

    # Train Diffusion Network
    net = UNetModel(image_size=256, in_channels=3, out_channels=3, 
                     model_channels=256, num_res_blocks=2, channel_mult=(1, 1, 2, 2, 4, 4),
                     attention_resolutions=[32,16,8], num_head_channels=64, dropout=0.1, resblock_updown=True, use_scale_shift_norm=True).to(device)
    net.train()

    print('Network parameters:', sum([p.numel() for p in net.parameters()]))

    class Dataset(Dataset):
            def __init__(
                self,
                image_folder,
                image_size,
                exts = ['jpg', 'jpeg', 'png', 'tiff'],
                augment_horizontal_flip = True
            ):
                super().__init__()
                self.image_folder=image_folder
                self.image_size = image_size
                self.image_paths = [p for ext in exts for p in Path(f'{image_folder}').glob(f'**/*.{ext}')]


                self.transform = T.Compose([
                    #T.CenterCrop(450),
                    T.Resize(256),
                    T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, index):
                image_path = self.image_paths[index]
                image = Image.open(image_path)
                return self.transform(image)

    ds = Dataset(image_folder, image_size, augment_horizontal_flip = augment_horizontal_flip)
    dl = DataLoader(ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 6)

    opt = torch.optim.Adam(net.parameters(), lr=1e-4,betas=(0.9,0.99))
    diffusion = GaussianDiffusion(T=1000, schedule='linear')

    with torch.no_grad():
        net.eval()
        x = diffusion.inverse(net, shape=(3, 256, 256), device=device)
        image=x.cpu().numpy()[0]
        np.clip(image,-1,1,out=image)
        image=image.transpose(1,2,0)
        #print(image.shape)
        image = Image.fromarray(np.uint8((image + 1) / 2 * 255))
        image.save('path/to/save/samples')
        net.train()

    epochs = 40
    update_every = 20
    accumulation_every=8
    for e in range(epochs):
        print(f'Epoch [{e + 1}/{epochs}]')
        losses = []
        batch_bar = tqdm.tqdm(dl)
        for i, img in enumerate(batch_bar):
            # Sample from the diffusion process
            t = np.random.randint(1, diffusion.T + 1, img.shape[0]).astype(int)
            xt, epsilon = diffusion.sample(img, t)  # q_sample
            t = torch.from_numpy(t).float().view(img.shape[0])
            # Pass through network
            out = net(xt.float().to(device), t.to(device))
            # Compute loss and backprop
            loss = F.mse_loss(out, epsilon.float().to(device))
            loss = loss / accumulation_every    # 16 img or 4 batch
            loss.backward()

            if (i+1) % accumulation_every == 0:
                opt.step()
                opt.zero_grad()

            #opt.step()

            losses.append(loss.item())
            if i % update_every == 0:

                batch_bar.set_postfix({'Loss': np.mean(losses)})
                losses = []

        batch_bar.set_postfix({'Loss': np.mean(losses)})
        losses = []

        # Save/Load model
        torch.save(net.state_dict(), 'path/to/save/models/isic18_unet-' + str(e + 1) + '.pth')
        print('Saved model')
        '''
        # save sample
        with torch.no_grad():
            net.eval()
            x = diffusion.inverse(net, shape=(3, 256, 256), device=device)
            image=x.cpu().numpy()[0]
            np.clip(image,-1,1,out=image)
            image=image.transpose(1,2,0)
            #print(image.shape)
            image = Image.fromarray(np.uint8((image + 1) / 2 * 255))
            image.save('/home/ylindq/YUANQIANSHI/CODE/diffusion_priors-main/isic/samples/sample'+str(e+1)+'.png')
            net.train()
        '''