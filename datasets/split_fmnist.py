from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import numpy as np
import torch
import torchvision


class SplitFMNIST(object):
    def __init__(self, input_size = 32):
        torch.manual_seed(1)
        np.random.seed(1)

        self.n_classes = 10

        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (.5,), (.5,)
            ),
        ])
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root = '../data',
            train = True,
            download = True,
            transform = transform
        )
        self.val_dataset = torchvision.datasets.FashionMNIST(
            root = '../data',
            train = True,
            download = True,
            transform = transform
        )
        n_train_samples = len(self.train_dataset)
        indices = list(range(n_train_samples))
        split = int(np.floor(.2 * n_train_samples))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root = '../data',
            train = False,
            download = True,
            transform = transform
        )

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = 64,
            num_workers = 2,
            drop_last = True,
            sampler = self.train_sampler
        )

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size = 64,
            num_workers = 2,
            drop_last = True,
            sampler = self.val_sampler
        )

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = 16,
            num_workers = 2,
            drop_last = False
        )

