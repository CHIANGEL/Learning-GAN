
# coding: utf-8

# # DCGAN
# 
# This is an implementation of DCGAN towards MNIST dataset using the GPU resources in lab.

# ### Setups for DCGAN

# In[1]:


# Set imports

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from IPython.display import HTML


# In[2]:


# Set random seed

manualSeed = 999
print('Random Seed: ', manualSeed)

random.seed(manualSeed)
torch.manual_seed(manualSeed)


# In[3]:


# Set image print function

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 


# In[4]:


# Root directory for dataset
dataroot = "./dataset"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# In[5]:


# Load dataset MNIST

dataset = dset.MNIST(root=dataroot, download=True,
                     transform=transforms.Compose([
                         transforms.Resize(image_size),
                         transforms.ToTensor(),
                     ]))
print('Data download successfully!')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=True, num_workers=workers)
print('Data loaded successfully!')

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('Set device successfully!')

imgs = dataloader.__iter__().next()[0].view(batch_size, 4096).numpy().squeeze()
print('Make imgs successfully!')

show_images(imgs)


# In[6]:


# Set weight initialization function for both CONV and BatchNorm
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ### Generator Code

# In[7]:


# Define a Generator class

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input Z size [100, 1, 1]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size [ngf * 8, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size [ngf * 4, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size [ngf * 2, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size [ngf, 32, 32]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # WARNING: Do not apply BatchNorm to the output layer of Generator! - DCGAN guideline
            nn.Tanh()
            # state size [nc, 64, 64]
        )
    def forward(self, input):
        return self.main(input)


# In[8]:


# Create the Generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Weight initialization
netG.apply(weights_init)

print(netG)


# ### Discriminator Code

# In[9]:


# Define a Generator class

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main =nn.Sequential(
            # input G(Z) size [nc, 64, 64]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # WARNING: Do not apply BatchNorm to the input layer of Discriminator! - DCGAN guideline
            nn.LeakyReLU(0.2, inplace=True),
            # state size [ndf, 32, 32]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size [ndf * 2, 16, 16]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size [ndf * 4, 8, 8]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size [ndf * 8, 4, 4]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size [1, 1, 1]
        )
    def forward(self, input):
        return self.main(input)


# In[10]:


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Weight initialzation
netD.apply(weights_init)

print(netD)


# ### Optimizer and Loss

# In[11]:


# Initialize BCE loss function
criterion = nn.BCELoss()

# Create a batch of latent vectors that we will use to visualize
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

print('fixed_noise: ', fixed_noise.size())

# Establish convention for real and fake labels
real_label = 1
fake_label = 0

# Setup Adam optimizer for G and D
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


# ### Training loop

# In[12]:


# List to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print('Starting Training Loop...')

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        # data is list containing 2 elements. first is the real input image, second is the corresponding label number
        real_cpu = data[0].to(device) 
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item() # item() transfer a Tenser number to a real number
        
        ## Train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1) # detach() remove a node from the computational graph
        # Calculate loss on all fake batch
        errD_fake = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        ## Add gradients from real & fake 
        errD = errD_real + errD_fake
        
        ## Update D
        optimizerD.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        netG.zero_grad()
        label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calcualte the G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
        ############################
        # (3) Output and Save training states
        ############################
        ## Output training states
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        ## Save loss of G and D
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        ## Check the outcome of G on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            # torch.save(netG, './saved_models/netG.pt')
            # torch.save(netD, './saved_models/netD.pt')
        iters += 1


# In[13]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[26]:


img_list_1 = img_list[0: 11]
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list_1]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# In[ ]:




