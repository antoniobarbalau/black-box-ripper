import numpy as np
import torch


class RandomFromGenerator(object):
    def __init__(
        self,
        generator,
        teacher,
        student,
        test_dataloader,
        soft_labels = True,
        device = torch.device('cuda'),
        to_grayscale = True,
        input_size = 32
    ):
        self.generator = generator
        self.teacher = teacher
        self.device = device
        self.test_dataloader = test_dataloader
        self.use_soft_labels = soft_labels
        self.to_grayscale = to_grayscale

    def train_dataloader(self, *a, **b):
        encoding_size = 128 if 'sngan' in str(type(self.generator)) else 256
        for _ in range(1000):
            images = self.generator(
                torch.Tensor(
                    # np.random.normal(size = (64, 256))
                    np.random.uniform(-3.3, 3.3, size = (64, encoding_size))
                ).cuda()
            )

            if self.to_grayscale:
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(self.device)
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

            with torch.no_grad():
                labels = self.teacher(images)
                if not self.use_soft_labels:
                    labels = labels.max(1)[1]
                else:
                    labels = torch.softmax(labels, dim = -1)
            yield (images, labels)


