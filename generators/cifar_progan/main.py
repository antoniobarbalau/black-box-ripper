from discriminator import Discriminator
from generator import Generator
from initializer import initializer
from interpolation_step import interpolation_step
from step import step
import numpy as np
import torch

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

dataroot = './data'

generator = Generator(ngpu = 1).to(device)
discriminator = Discriminator(ngpu = 1).to(device)
inference_generator = Generator(ngpu = 1).to(device)

def accumulate(inference_generator, generator, momentum = .999):
    inference_parameters = dict(inference_generator.named_parameters())
    parameters = dict(generator.named_parameters())

    for parameter_name in inference_parameters.keys():
        inference_parameters[parameter_name].data.mul_(
            momentum
        ).add_(
            1. - momentum, parameters[parameter_name].data
        )
accumulate(inference_generator, generator, 0.)

discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr = .01,
    betas = (.0, .99)
)
generator_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr = .01,
    betas = (.0, .99)
)
with torch.no_grad():
    test_noise = torch.Tensor(
        # np.random.normal(size = (256, 256))
        np.random.uniform(-5., 5., size = (256, 256))
    ).to(device)

step(
    4, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_1, inference_generator.generate_1,
    discriminator.discriminate_1,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)

interpolation_step(
    8, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_2, inference_generator.generate_2,
    discriminator.discriminate_2,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
step(
    8, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_3, inference_generator.generate_3,
    discriminator.discriminate_3,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
# torch.save(generator.state_dict(), f'./split_generator')
# torch.save(discriminator.state_dict(), f'./split_discriminator')
# torch.save(inference_generator.state_dict(), f'./split_inference_generator')

interpolation_step(
    16, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_4, inference_generator.generate_4,
    discriminator.discriminate_4,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
step(
    16, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_5, inference_generator.generate_5,
    discriminator.discriminate_5,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
# torch.save(generator.state_dict(), f'./split_generator')
# torch.save(discriminator.state_dict(), f'./split_discriminator')
# torch.save(inference_generator.state_dict(), f'./split_inference_generator')

interpolation_step(
    32, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_6, inference_generator.generate_6,
    discriminator.discriminate_6,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
step(
    32, 400, test_noise,
    generator, inference_generator, discriminator, accumulate,
    generator.generate_7, inference_generator.generate_7,
    discriminator.discriminate_7,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
# torch.save(generator.state_dict(), f'./split_generator')
# torch.save(discriminator.state_dict(), f'./split_discriminator')
# torch.save(inference_generator.state_dict(), f'./split_inference_generator')

# generator.load_state_dict(torch.load('./generator'))
# discriminator.load_state_dict(torch.load('./discriminator'))

# interpolation_step(
#     48, 400, test_noise,
#     generator, inference_generator, discriminator, accumulate,
#     generator.generate_8, inference_generator.generate_8,
#     discriminator.discriminate_8,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# step(
#     48, 400, test_noise,
#     generator, inference_generator, discriminator, accumulate,
#     generator.generate_9, inference_generator.generate_9,
#     discriminator.discriminate_9,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
torch.save(generator.state_dict(), f'./split_90_generator_final')
torch.save(discriminator.state_dict(), f'./split_90_discriminator_final')
torch.save(inference_generator.state_dict(), f'./split_90_inference_generator_final')

