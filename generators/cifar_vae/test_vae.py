from vae import VAE
import argparse
import cv2
import numpy as np
import os
import shutil
import torch
import torchvision


arg_parser = argparse.ArgumentParser(description = 'Testing hyperparameters')

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

vae = VAE(env.grayscale).cuda()
colorscheme = 'grayscale' if env.grayscale else 'rgb'
vae.load_state_dict(torch.load(f'./cifar_{colorscheme}_vae_state_dict'))
vae.train(False)

n_channels = 1 if env.grayscale else 3
test_images = torchvision.datasets.CIFAR10(
    root = env.dataset_path,
    train = False,
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

test_dataloader = torch.utils.data.DataLoader(
    test_images,
    batch_size = 1,
    shuffle = True,
    num_workers = 2,
    drop_last = True
)

output_folder = f'{colorscheme}_reconstructions'
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)
for index, image in enumerate(test_dataloader):
    image = image[0].cuda()

    _, _, _, reconstructed_image = vae(image)
    reconstructed_image = reconstructed_image.clamp(-1, 1)
    image = image.clamp(-1, 1)

    image = image.detach().cpu().numpy()
    reconstructed_image = reconstructed_image.detach().cpu().numpy()
    image = np.concatenate([image, reconstructed_image], axis = 3)
    image = np.transpose(image, [0, 2, 3, 1])[0]
    image = image / 2. + .5
    image = np.uint8(image * 255.)

    cv2.imwrite(
        os.path.join(output_folder, f'{index}.png'),
        image
    )

