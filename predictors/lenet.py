import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, name, n_outputs):
        super().__init__()

        self.name = name

        self.conv1 = nn.Conv2d(
                in_channels = 1,
                out_channels = 6,
                kernel_size = 5,
                stride = 1,
                padding = 0,
            )
        self.conv1.bias.data.normal_(0, 0.1)
        self.conv1.bias.data.fill_(0)

        self.max_pool_1 = nn.MaxPool2d(kernel_size = 2)

        self.conv2 = nn.Conv2d(
                in_channels = 6,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding = 0,
            )
        self.conv2.bias.data.normal_(0, 0.1)
        self.conv2.bias.data.fill_(0)

        self.max_pool_2 = nn.MaxPool2d(kernel_size = 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1.bias.data.normal_(0, 0.1)
        self.fc1.bias.data.fill_(0)

        self.fc2 = nn.Linear(120, 84)
        self.fc2.bias.data.normal_(0, 0.1)
        self.fc2.bias.data.fill_(0)

        self.fc3 = nn.Linear(84, n_outputs)
        self.fc3.bias.data.normal_(0, 0.1)
        self.fc3.bias.data.fill_(0)


    def forward(self, input):
        x = torch.relu(self.conv1(input))
        x = self.max_pool_1(x)
        x = torch.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

