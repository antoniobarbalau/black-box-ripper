import torch
import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, name, n_outputs):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = models.resnet18(pretrained = False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        input_size = 224

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
