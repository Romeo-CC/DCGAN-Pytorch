
import torch
import torch.nn as nn
import torch.optim as optim

from models.gan import Generator, Discriminator, weights_init
from metrics.loss import binary_cross_entropy

from utils.dataset import LmdbDataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as tfms

import torchvision.utils as vutils

import json

from tqdm import tqdm

from time import localtime, strftime
import numpy as np
import os
import hashlib

import matplotlib.pyplot as plt

train_cfg_path = "configs/train.json"
with open(train_cfg_path) as rf:
    cfg_txt = rf.read()
train_cfg = json.loads(cfg_txt)

batch_size = train_cfg["batch_size"]
learning_rate = train_cfg["learning_rate"]
beta1 = train_cfg["beta1"]
ngpu = train_cfg["ngpu"]
available_gpus = train_cfg["available_gpus"]
num_epochs = train_cfg["num_epochs"]
workers = train_cfg["workers"]

device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")

model_cfg_path = "configs/gan.json"
with open(model_cfg_path) as rf:
    cfg_txt = rf.read()
model_cfg = json.loads(cfg_txt)

channels = model_cfg["nc"] # imgs channels
gen_feature_size = model_cfg["ngf"] # generator feature map size
dis_feature_size = model_cfg["ndf"] # discriminator feature map size
latent_dim = model_cfg["nz"] # latent vector dimension



generator = Generator(latent_dim, gen_feature_size, channels).to(device)

discriminator = Discriminator(channels, dis_feature_size).to(device)

if (device.type == "cuda") and (ngpu > 1):
    generator = nn.DataParallel(generator, available_gpus)
    discriminator = nn.DataParallel(discriminator, available_gpus)
    
generator.apply(weights_init)

discriminator.apply(weights_init)

print(generator)
print(discriminator)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(discriminator.parameters(), lr = learning_rate, betas = (beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr = learning_rate, betas = (beta1, 0.999))


train_data_path = os.path.join("data", "celeba")
dataset = LmdbDataset(train_data_path, "gen", transforms = [tfms.ToTensor(), tfms.Resize((64, 64)), tfms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

total_iterations = len(dataloader)

# set tensorboard writer


def time_stamp(format_str = "%Y_%m_%d %H:%M%S"):
    stamp = strftime(format_str, localtime())
    md5 = hashlib.md5(stamp.encode("utf-8")).hexdigest()
    return stamp, md5

log_stamp, encrypt = time_stamp("%Y_%m_%d")
log_path = os.path.join("tensorboard", f"{log_stamp}_{encrypt}")
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_path)




print("Starting Training First GAN...")

for epoch in range(num_epochs):
    # For each batch in the dataloader
    pbar = tqdm(enumerate(dataloader), total = total_iterations)
    for i, data in pbar:
        pbar.set_description(f"Now iter batch {i} in epoch {epoch + 1}")
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_imgs = data.to(device)
        b_size = real_imgs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_imgs).view(-1)
        # Calculate loss on all-real batch
        errD_real = binary_cross_entropy(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = binary_cross_entropy(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = binary_cross_entropy(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 100 == 0:
            print("[epoch: %d/%d][iter: %d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                  % (epoch + 1, num_epochs, i, total_iterations,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())
        # log_to_tensorboard
        iters = total_iterations * epoch + i
        # if iters + 1 % 10 == 0:
        writer.add_scalars("train_dcgan", {"Loss_D": errD.item(), "Loss_G": errG.item()}, iters)

        # Check how the generator is doing by saving G's output on fixed_noise
        
        if (i + 1) % 1000 == 0 or i == total_iterations - 1:
            generator.eval()
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                fake_imgs = np.transpose(vutils.make_grid(fake, padding = 2, normalize = True), (1, 2, 0))
                fake_name = os.path.join("data", "fake_samples", f"epoch_{epoch}_iter_{i}.png")
                plt.imsave(fake_name, fake_imgs.numpy())
            generator.train()

    # save model weight
    _, codec = time_stamp()
    dis_name = os.path.join("model_weights", "discriminator", f"discriminator_epoch_{epoch}_{codec}.pth")
    gen_name = os.path.join("model_weights", "generator", f"generator_epoch_{epoch}_{codec}.pth")

    torch.save(generator.state_dict(), gen_name)
    torch.save(discriminator.state_dict(), dis_name)

writer.close()