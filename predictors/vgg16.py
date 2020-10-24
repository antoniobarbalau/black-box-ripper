from torch import nn
import torch

class VGG(nn.Module):
    def __init__(self, name, n_channels = 1, n_outputs = 10):
        super(VGG, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'
        ])
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)


