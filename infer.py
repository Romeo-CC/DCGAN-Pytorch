import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from models.gan import Generator

import numpy as np
import json, os

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

with open("configs/gan.json") as rf:
    gan_cfg = rf.read()

model_cfg = json.loads(gan_cfg)

channels = model_cfg["nc"] # imgs channels
gen_feature_size = model_cfg["ngf"] # generator feature map size
latent_dim = model_cfg["nz"] # latent vector dimension

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


weight_path = "model_weights/generator/generator_epoch_35_74f322227719a5469fb7de2b10e8bc8d.pth"
generator = Generator(latent_dim, gen_feature_size, channels)

weight = torch.load(weight_path)
generator.load_state_dict(weight)
generator.to(device)
generator.eval()

noise = torch.randn(64, latent_dim, 1, 1, device=device)

with torch.no_grad():
    fakes = generator(noise).detach().cpu()

plt.figure("generated fakes", figsize=(8, 8))
plt.axis("off")
fake_imgs = np.transpose(vutils.make_grid(fakes, padding = 2, normalize = True), (1, 2, 0))
plt.imshow(fake_imgs)
plt.imsave(os.path.join("data", "fake_samples", "gen2.png"), fake_imgs.numpy())
plt.show()