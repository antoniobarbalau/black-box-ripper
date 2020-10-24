from functools import reduce
from .layers import *
import torch

class Generator(nn.Module):
    def __init__(self, grayscale = True):
        super(Generator, self).__init__()

        self.from_latent = torch.nn.Sequential(
            nn.Linear(256, 4 * 4 * 256),
            Reshape(-1, 256, 4, 4),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),

            EqualConv2d(
                in_channels = 256,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )

        self.conv_blocks = [
            [
                torch.nn.Upsample(
                    scale_factor = 2,
                    mode = 'bilinear'
                ),
                torch.nn.Conv2d(
                    in_channels = filters * 2,
                    out_channels = filters,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                PixelNormalization(),
                torch.nn.LeakyReLU(negative_slope = 0.2),
                torch.nn.Conv2d(
                    in_channels = filters,
                    out_channels = filters,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                PixelNormalization(),
                torch.nn.LeakyReLU(negative_slope = 0.2),
            ]
            for filters in [128, 64, 32]
        ]
        self.conv_blocks = reduce(
            lambda list_0, list_1: list_0 + list_1,
            self.conv_blocks
        )
        self.conv_blocks = torch.nn.Sequential(*self.conv_blocks)

        self.to_rgb = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels = 32,
                out_channels = 1 if grayscale else 3,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )

    def forward(self, samples):
        return self.to_rgb(
            self.conv_blocks(
                self.from_latent(
                    samples
                )
            )
        )

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, samples):
        return samples.view(self.shape)

class PixelNormalization(nn.Module):
    def __init__(self):
        super(PixelNormalization, self).__init__()

    def forward(self, output):
        return output / torch.sqrt(
            torch.unsqueeze(
                torch.mean(
                    torch.pow(output, 2),
                    dim = [1],
                ) + 1e-8,
                1
            )
        )

