import torch
from torch import nn
from plot_samples import plot_samples
import torchvision.transforms as transforms
import torchvision
import numpy as np

def interpolation_step(
    image_size, n_epochs, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator_function, inference_generator_function, discriminator_function,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
):
    # allowed = [
    #     'mushroom', 'house', 'sunflower', 'tulip', 'mountain',
    #     'orchid', 'bed', 'castle', 'table', 'palm_tree'
    # ]

    vehicles = 'bicycle, bus, motorcycle, pickup_truck, train, lawn_mower, rocket, streetcar, tank, tractor'
    allowed = vehicles.replace(' ', '').split(',')
    dataset = torchvision.datasets.CIFAR100(
        root = './data',
        train = True,
        download = True,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            # torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [.5], [.5]
            )
        ])
    )
    allowed = [
        dataset.class_to_idx[a]
        for a in allowed
    ]
    indexes = [i for i, value in enumerate(dataset.targets) if value not in allowed]
    dataset.data = dataset.data[indexes]
    dataset.targets = [dataset.targets[i] for i in indexes]

    for epoch in range(n_epochs):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 256,
            shuffle = True,
            num_workers = 2,
            drop_last = True
        )
        n_iterations = len(dataloader)
        for iter_n, samples in enumerate(dataloader):
            print(
                f'Size: {image_size} - Epoch: {epoch + 1}/{n_epochs}' +
                f' - Done: {np.round(iter_n / n_iterations, 2)}',
                end = '\r'
            )
            alpha = torch.Tensor((
                (iter_n + (epoch * n_iterations) ) /
                (n_epochs * n_iterations),
            )).to(device)
            samples = samples[0].to(device)
            samples = torch.nn.Upsample(scale_factor = 2, mode = 'bilinear')(
                torch.nn.AvgPool2d(2)(samples)
            ) * (1. - alpha) + samples * alpha

            generator.zero_grad()
            discriminator.zero_grad()

            discriminator_samples = discriminator_function(samples, alpha)
            generated_samples = generator_function(
                torch.Tensor(
                    # np.random.normal(size = (256, 256))
                    np.random.uniform(-5., 5., size = (256, 256))
                ).to(device),
                alpha
            )
            discriminator_generated = discriminator_function(generated_samples, alpha)
            x_hat_alpha = torch.Tensor(
                np.random.uniform(size = (1,))
            ).to(device)
            x_hat = x_hat_alpha * samples + (1. - x_hat_alpha) * generated_samples
            gradient = torch.autograd.grad(
                outputs = torch.mean(discriminator_function(x_hat, alpha)),
                inputs = x_hat
            )[0]
            gradient = torch.pow(torch.sqrt(
                torch.sum(
                    torch.pow(gradient, 2),
                    dim = [1, 2, 3]
                )
            ) - 1., 2)
            discriminator_loss = torch.mean(
                discriminator_generated - discriminator_samples +
                10. * gradient +
                .005 * torch.pow(discriminator_generated, 2) +
                .005 * torch.pow(discriminator_samples, 2)
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator.zero_grad()
            discriminator.zero_grad()
            generated_samples = generator_function(
                torch.Tensor(
                    # np.random.normal(size = (256, 256))
                    np.random.uniform(-5., 5., size = (256, 256))
                ).to(device),
                alpha
            )
            discriminator_generated = discriminator_function(generated_samples, alpha)
            generator_loss = torch.mean(-1. * discriminator_generated)
            generator_loss.backward()
            generator_optimizer.step()

            accumulate(inference_generator, generator)

        with torch.no_grad():
            generated_samples = inference_generator_function(test_noise, alpha)
            plot_samples(
                generated_samples,
                save = f'{image_size}_{epoch}'
            )
        # with torch.no_grad():
        #     generated_samples = generator_function(test_noise)
        #     plot_samples(generated_samples)
        # torch.save(generator.state_dict(), f'./generator_{epoch}')
        # torch.save(discriminator.state_dict(), f'./discriminator_{epoch}')


