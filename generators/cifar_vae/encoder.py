from functools import reduce
from .layers import *
import torch

class Encoder(nn.Module):
    def __init__(self, grayscale = False):
        super(Encoder, self).__init__()

        self.from_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 1 if grayscale else 3,
                out_channels = 32,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
            nn.LeakyReLU(negative_slope = 0.2)
        )

        self.conv_blocks = [
            [
                EqualConv2d(
                    in_channels = filters,
                    out_channels = filters,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                nn.LeakyReLU(negative_slope = 0.2),
                EqualConv2d(
                    in_channels = filters,
                    out_channels = 2 * filters,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1
                ),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.MaxPool2d(2)
            ]
            for filters in [32, 64, 128]
        ]
        self.conv_blocks = reduce(
            lambda list_0, list_1: list_0 + list_1,
            self.conv_blocks
        )
        self.conv_blocks = torch.nn.Sequential(*self.conv_blocks)

        self.to_latent = torch.nn.Sequential(
            EqualConv2d(
                in_channels = 256,
                out_channels = 256,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            torch.nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 256,
                out_channels = 512,
                kernel_size = 4,
                stride = 1,
                padding = 0
            ),
            torch.nn.LeakyReLU(negative_slope = 0.2),
            torch.nn.Flatten(),

            EqualLinear(in_features = 512, out_features = 512)
        )

    def forward(self, samples):
        return self.to_latent(
            self.conv_blocks(
                self.from_rgb(
                    samples
                )
            )
        )

