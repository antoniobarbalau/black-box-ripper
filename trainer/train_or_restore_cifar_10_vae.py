import torch
import setup

def train_or_restore_cifar_10_vae(generator, device = torch.device('cuda')):
    generator.load_state_dict(torch.load(
        './checkpoints/cifar_10_grayscale_vae',
        map_location = setup.device
    ))
    generator.to(setup.device)
    generator.eval()

    return generator.generate
