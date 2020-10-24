import torch
import setup

def train_or_restore_cifar_100_10_classes_gan(generator):
    generator.load_state_dict(torch.load(
        './checkpoints/cifar_100_10_classes_gan',
        map_location = setup.device
    ))
    generator.to(setup.device)
    generator.eval()

    return generator

