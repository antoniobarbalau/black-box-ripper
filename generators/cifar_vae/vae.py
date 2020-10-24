import torch
from .generator import Generator
from .encoder import Encoder
from torch import nn


class VAE(nn.Module):
    def __init__(self, grayscale = True):
        super(VAE, self).__init__()

        self.generator = Generator(grayscale)
        self.encoder = Encoder(grayscale)

    def encoding_size(self):
        return 256

    def sample(self, encoder_output):
        means = encoder_output[:, :256]
        stds = encoder_output[:, 256:]
        return (
            means,
            stds,
            means + torch.exp(.5 * stds) * torch.randn(
                size = (encoder_output.size(0), 256),
                device = 'cuda'
            )
        )

    def generate(self, noise):
        return self.generator(noise)

    def forward(self, inputs):
        means, stds, encodings = self.sample(self.encoder(inputs))
        reconstructions = self.generator(encodings)

        return means, stds, encodings, reconstructions

