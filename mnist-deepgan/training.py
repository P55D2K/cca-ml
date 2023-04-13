print("\033cDEEPGAN WITH MNIST DATASET | TRAINING\nOriginal Script by Darrel\nModified by ZK\n")

import argparse
import os, sys
import numpy as np
import math, json

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from cmodels.generator import Generator
from cmodels.discriminator import Discriminator

currentdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(currentdir)

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()

def get_args(message):
    use_defaults = input(message)
    if use_defaults == "y":
        args = {
            "--n_epochs": "200",
            "--batch_size": "64",
            "--lr": "0.0002",
            "--b1": "0.5",
            "--b2": "0.999",
            "--n_cpu": "8",
            "--latent_dim": "100",
            "--img_size": "28",
            "--channels": "1",
            "--sample_interval": "400"
        }
    else:
        n_epoches = input("n_epochs (default 200): ")
        batch_size = input("batch_size (default 64): ")
        lr = input("lr (default 0.0002): ")
        b1 = input("b1 (default 0.5): ")
        b2 = input("b2 (default 0.999): ")
        n_cpu = input("n_cpu (default 8): ")
        latent_dim = input("latent_dim (default 100): ")
        img_size = input("img_size (default 28): ")
        channels = input("channels (default 1): ")
        sample_interval = input("sample_interval (default 400): ")
        args = {
            "--n_epochs": n_epoches,
            "--batch_size": batch_size,
            "--lr": lr,
            "--b1": b1,
            "--b2": b2,
            "--n_cpu": n_cpu,
            "--latent_dim": latent_dim,
            "--img_size": img_size,
            "--channels": channels,
            "--sample_interval": sample_interval
        }
    with open("config.json", "w") as f:
        f.write(json.dumps(args))
    return args

if os.path.isfile("config.json"):
    with open("config.json", "r") as f:
        file_contents = f.read()
        print("config.json file found. Using values from file.")
        if len(file_contents) > 0:
            args = json.loads(file_contents)
        else:
            args = get_args("No values found in config.json. Use default values? (y/n): ")
else:
    args = get_args("No config.json file found. Use default values? (y/n): ")

parser.add_argument("--n_epochs", type=int, default=int(args["--n_epochs"]), help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=int(args["--batch_size"]), help="size of the batches")
parser.add_argument("--lr", type=float, default=float(args["--lr"]), help="adam: learning rate")
parser.add_argument("--b1", type=float, default=float(args["--b1"]), help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=float(args["--b2"]), help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=int(args["--n_cpu"]), help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=int(args["--latent_dim"]), help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=int(args["--img_size"]), help="size of each image dimension")
parser.add_argument("--channels", type=int, default=int(args["--channels"]), help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=int(args["--sample_interval"]), help="interval betwen image samples")

opt = parser.parse_args()
print(opt); print()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt, img_shape)
discriminator = Discriminator(img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

model_name = input("Enter model name: ")
os.makedirs(("images/" + model_name + "/"), exist_ok=True)

num_epochs = args["--n_epochs"]
for epoch in range(int(num_epochs)):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % ((epoch+1), int(num_epochs), (i+1), len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        image_format_batches_done = "epoch_" + str(epoch+1) + "_batch_" + str(i+1)
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], ("images/" + model_name + "/" + image_format_batches_done + ".png"), nrow=5, normalize=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer_G.state_dict(),
        'loss': g_loss,
    }, "saved_models/" + model_name + ".pth")
