from torch import nn
import torch
from .eqalized_layers import *

class Generator(nn.Module):
    def __init__(self, ngpu = 1):
        super(Generator, self).__init__()

        self.block_1 = nn.Sequential(
            EqualLinear(256, 4 * 4 * 256),
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
        self.block_1_to_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 256,
                out_channels = 3,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_2 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            EqualConv2d(
                in_channels = 256,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 128,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_2_to_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 128,
                out_channels = 3,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_3 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            EqualConv2d(
                in_channels = 128,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 64,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_3_to_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 64,
                out_channels = 3,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_4 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            EqualConv2d(
                in_channels = 64,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_4_to_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 32,
                out_channels = 3,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.block_5 = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'bilinear'
            ),
            EqualConv2d(
                in_channels = 32,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
            EqualConv2d(
                in_channels = 16,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            PixelNormalization(),
            nn.LeakyReLU(negative_slope = 0.2),
        )
        self.block_5_to_rgb = nn.Sequential(
            EqualConv2d(
                in_channels = 16,
                out_channels = 3,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )
        )
        self.ngpu = ngpu
        self.upsample = nn.Upsample(
            scale_factor = 2,
            mode = 'bilinear'
        )

    def generate_1(self, input):
        return self.block_1_to_rgb(self.block_1(input))

    def generate_2(self, samples, alpha):
        old_output = self.block_1(samples)

        output = self.block_2_to_rgb(self.block_2(old_output))
        old_output = self.block_1_to_rgb(old_output)
        old_output = self.upsample(old_output)

        return old_output * (1. - alpha) + output * alpha

    def generate_3(self, samples):
        return self.block_2_to_rgb(self.block_2(self.block_1(samples)))

    def generate_4(self, samples, alpha):
        old_output = self.block_2(self.block_1(samples))

        output = self.block_3_to_rgb(self.block_3(old_output))
        old_output = self.block_2_to_rgb(old_output)
        old_output = self.upsample(old_output)

        return old_output * (1. - alpha) + output * alpha

    def generate_5(self, samples):
        old_output = self.block_2(self.block_1(samples))
        output = self.block_3_to_rgb(self.block_3(old_output))

        return output

    def generate_6(self, samples, alpha):
        old_output = self.block_3(self.block_2(self.block_1(samples)))

        output = self.block_4_to_rgb(self.block_4(old_output))
        old_output = self.block_3_to_rgb(old_output)
        old_output = self.upsample(old_output)

        return old_output * (1. - alpha) + output * alpha

    def generate_7(self, samples):
        old_output = self.block_3(self.block_2(self.block_1(samples)))
        output = self.block_4_to_rgb(self.block_4(old_output))

        return output

    def generate_8(self, samples, alpha):
        old_output = self.block_4(self.block_3(self.block_2(self.block_1(samples))))

        output = self.block_5_to_rgb(self.block_5(old_output))
        old_output = self.block_4_to_rgb(old_output)
        old_output = self.upsample(old_output)

        return old_output * (1. - alpha) + output * alpha

    def generate_9(self, samples):
        old_output = self.block_4(self.block_3(self.block_2(self.block_1(samples))))
        output = self.block_5_to_rgb(self.block_5(old_output))

        return output

    def forward(self, input):
        return self.generate_7(input)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

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

