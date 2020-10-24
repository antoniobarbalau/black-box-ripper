import torch
from generator import Generator
from discriminator import Discriminator
from initializer import initializer
from interpolation_step import interpolation_step
from step import step

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

dataroot = '/home/perses/tonio/datasets/fer/data/'

generator = Generator(ngpu = 1).to(device)
discriminator = Discriminator(1).to(device)

# generator.apply(initializer)
# discriminator.apply(initializer)

discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr = .001,
    betas = (.0, .99)
)
generator_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr = .001,
    betas = (.0, .99)
)

# step(
#     3, 20,
#     generator, discriminator,
#     generator.generate_1, discriminator.discriminate_1,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )

# interpolation_step(
#     6, 20,
#     generator, discriminator,
#     generator.generate_2, discriminator.discriminate_2,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# step(
#     6, 20,
#     generator, discriminator,
#     generator.generate_3, discriminator.discriminate_3,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# torch.save(generator.state_dict(), f'./generator')
# torch.save(discriminator.state_dict(), f'./discriminator')

# interpolation_step(
#     12, 20,
#     generator, discriminator,
#     generator.generate_4, discriminator.discriminate_4,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# step(
#     12, 20,
#     generator, discriminator,
#     generator.generate_5, discriminator.discriminate_5,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# torch.save(generator.state_dict(), f'./generator')
# torch.save(discriminator.state_dict(), f'./discriminator')

# interpolation_step(
#     24, 20,
#     generator, discriminator,
#     generator.generate_6, discriminator.discriminate_6,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# step(
#     24, 20,
#     generator, discriminator,
#     generator.generate_7, discriminator.discriminate_7,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
# torch.save(generator.state_dict(), f'./generator')
# torch.save(discriminator.state_dict(), f'./discriminator')

generator.load_state_dict(torch.load('./generator'))
discriminator.load_state_dict(torch.load('./discriminator'))

# interpolation_step(
#     48, 40,
#     generator, discriminator,
#     generator.generate_8, discriminator.discriminate_8,
#     generator_optimizer, discriminator_optimizer,
#     dataroot, device
# )
step(
    48, 40,
    generator, discriminator,
    generator.generate_9, discriminator.discriminate_9,
    generator_optimizer, discriminator_optimizer,
    dataroot, device
)
# torch.save(generator.state_dict(), f'./generator_final')
# torch.save(discriminator.state_dict(), f'./discriminator_final')
