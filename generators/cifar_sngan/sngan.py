import torch
from .sngan_cifar10 import Generator


class SNGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Args():
            pass
        args = Args()
        args.latent_dim = 128
        args.gf_dim = 3
        args.bottom_width = 4
        self.generator = Generator(args = args)

    def forward(self, inputs):
        return self.generator(inputs)


