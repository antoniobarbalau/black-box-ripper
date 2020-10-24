from vae import VAE
import argparse
import cv2
import numpy as np
import torch
import torchvision

arg_parser = argparse.ArgumentParser(description = 'Training hyperparameters')

arg_parser.add_argument(
    '--dataset_path',
    type = str,
    default = '/home/tonio/research/datasets/cifar'
)

arg_parser.add_argument(
    '--grayscale',
    default = False,
    action = 'store_const',
    const = True
)

env = arg_parser.parse_args()

n_channels = 1 if env.grayscale else 3
train_images = torchvision.datasets.CIFAR10(
    root = env.dataset_path,
    train = True,
    download = True,
    transform = torchvision.transforms.Compose(
        ([torchvision.transforms.Grayscale()] if env.grayscale else []) + 
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [.5] * n_channels, [.5] * n_channels
            )
        ]
    )
) 

vae = VAE(
    grayscale = env.grayscale
).cuda()
optimizer = torch.optim.Adam(
    vae.parameters(),
    lr = 1e-3
)

n_epochs = 120
scheduler = [
    0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1., 1., 1., 1., 1.
] * 8

for epoch_n in range(n_epochs):
    print(f'Running epoch: {epoch_n} / {n_epochs}', end = '\r')
    beta = scheduler[epoch_n]
    train_dataloader = torch.utils.data.DataLoader(
        train_images,
        batch_size = 64,
        shuffle = True,
        num_workers = 2,
        drop_last = True
    )
    for iter_n, batch in enumerate(train_dataloader):
        images = batch[0].cuda()
        
        vae.zero_grad()
        means, stds, embeddings, reconstructions = vae(images)
        reconstruction_loss = torch.pow(images - reconstructions, 2).sum(
            axis = [1, 2, 3]
        )
        kl_loss = - 0.5 * torch.sum(
            1 + stds - torch.pow(means, 2) - torch.exp(stds),
            axis = -1
        )
        loss = torch.mean(reconstruction_loss + beta * kl_loss)
        loss.backward()
        optimizer.step()

colorscheme = 'grayscale' if env.grayscale else 'rgb'
torch.save(vae.state_dict(), f'./cifar_{colorscheme}_vae_state_dict')
