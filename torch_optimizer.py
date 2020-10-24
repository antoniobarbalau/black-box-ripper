import numpy as np
import torch
import setup


def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(
        softmax -
        np.eye(10)[label],
        2
    ))

def optimize_to_grayscale(classifier, generator, batch_size = 64, encoding_size = 128):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        label = np.random.randint(10, size = (1, 1))
        while c < .9 and x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().cuda())
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum((1,), keepdim = True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
            c = softmaxes[indexes[0]][label]
        batch.append(image)
    return torch.cat(batch)#, axis = 0)


def optimize_rescale(classifier, generator, batch_size = 16):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 512
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size = (20, encoding_size))
        label = np.random.randint(10, size = (1, 1))
        while c < .90 and x < 3:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float())
                images = torch.nn.functional.interpolate(images, size = 224)
                # multipliers = [.2126, .7152, .0722]
                # multipliers = np.expand_dims(multipliers, 0)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.tile(multipliers, [1, 1, 32, 32])
                # multipliers = torch.Tensor(multipliers).to(device)
                # images = images * multipliers
                # images = images.sum(axis = 1, keepdims = True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
            ])
            c = softmaxes[indexes[0]][label]
        batch.append(image)
    return torch.cat(batch)#, axis = 0)

def optimize(classifier, generator, batch_size = 64):
    batch = []

    n_iter = batch_size
    encoding_size =256
    if 'sngan' in str(type(generator)):
        encoding_size = 128
    for i in range(n_iter):
        c = 0.
        x = 0
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        label = np.random.randint(10, size = (1, 1))
        while c < .9 and x < 300:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to(setup.device))
                # multipliers = [.2126, .7152, .0722]
                # multipliers = np.expand_dims(multipliers, 0)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.expand_dims(multipliers, -1)
                # multipliers = np.tile(multipliers, [1, 1, 32, 32])
                # multipliers = torch.Tensor(multipliers).to(device)
                # images = images * multipliers
                # images = images.sum(axis = 1, keepdims = True)

                softmaxes = classifier(images).detach().cpu()
            losses = [loss(np.array(s), i, label) for s, i in zip(softmaxes, images)]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
            c = softmaxes[indexes[0]].numpy()
            c = c[label]
        batch.append(image)
    return torch.cat(batch)#, axis = 0)


def discrepancy_loss(teacher_predictions, student_predictions):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return -np.sum(np.square(teacher_softmax - student_softmax))


def optimize_discrepancies(teacher, student, generator, batch_size = 16):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        x = 0
        while x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to(setup.device))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                discrepancy_loss(np.array(s), np.array(i))
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis = 0)

def discrepancy_loss_kl(teacher_predictions, student_predictions):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return -np.sum(teacher_softmax * np.log(teacher_softmax) / np.log(student_softmax))

def optimize_discrepancies_kl(teacher, student, generator, batch_size = 64):
    batch = []

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        x = 0
        while x < 50:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().to(setup.device))
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                discrepancy_loss_kl(np.array(s), np.array(i))
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
        batch.append(image)
        print('image')
    return torch.cat(batch, axis = 0)


def loss(softmax, image, label):
    softmax = np.exp(softmax) / np.sum(np.exp(softmax))
    return np.sum(np.power(
        softmax -
        np.eye(10)[label],
        2
    ))

def optimize_discrepancies_(teacher, student, generator, batch_size = 64):
    encoding_size = 128
    batch_size = 16

    with torch.no_grad():
        specimens = torch.tensor(
            np.random.uniform(-3.3, 3.3, size = (batch_size, 30, encoding_size))
        ).float().to(setup.device)

        for _ in range(10):
            images = generator(specimens.view(-1, encoding_size))
            multipliers = [.2126, .7152, .0722]
            multipliers = np.expand_dims(multipliers, 0)
            multipliers = np.expand_dims(multipliers, -1)
            multipliers = np.expand_dims(multipliers, -1)
            multipliers = np.tile(multipliers, [1, 1, 32, 32])
            multipliers = torch.Tensor(multipliers).to(setup.device)
            images = images * multipliers
            images = images.sum(axis = 1, keepdims = True)
            teacher_predictions = torch.softmax(
                teacher(images), axis = -1
            ).detach().cpu()
            student_predictions = torch.softmax(
                student(images), axis = -1
            ).detach().cpu()

            losses = -1. * torch.pow(
                teacher_predictions - student_predictions, 2
            ).sum(-1).view(batch_size, 30)
            indexes = torch.argsort(losses) < 10
            specimens = specimens[indexes].view(batch_size, 10, encoding_size)
            specimens = torch.cat((
                specimens,
                specimens + torch.randn(batch_size, 10, encoding_size).to(setup.device),
                specimens + torch.randn(batch_size, 10, encoding_size).to(setup.device),
            ), axis = 1)

        images = generator(specimens.view(-1, encoding_size))
        images = generator(specimens.view(-1, encoding_size))
        multipliers = [.2126, .7152, .0722]
        multipliers = np.expand_dims(multipliers, 0)
        multipliers = np.expand_dims(multipliers, -1)
        multipliers = np.expand_dims(multipliers, -1)
        multipliers = np.tile(multipliers, [1, 1, 32, 32])
        multipliers = torch.Tensor(multipliers).to(setup.device)
        images = images * multipliers
        images = images.sum(axis = 1, keepdims = True)

        teacher_predictions = torch.softmax(
            teacher(images), axis = -1
        ).detach().cpu()
        student_predictions = torch.softmax(
            student(images), axis = -1
        ).detach().cpu()

        losses = -1. * torch.pow(
            teacher_predictions - student_predictions, 2
        ).sum(-1).view(batch_size, 30)

        indexes = torch.argsort(losses) < 1
        images = images.view(batch_size, 30, 1, 32, 32)[indexes]
    return images




def curriculum_loss(teacher_predictions, student_predictions, label, weight):
    teacher_softmax = np.exp(teacher_predictions) / np.sum(
        np.exp(teacher_predictions)
    )
    student_softmax = np.exp(student_predictions) / np.sum(
        np.exp(student_predictions)
    )
    return (
        np.sum(np.square(teacher_softmax - label)) -
        weight * np.sum(np.square(teacher_softmax - student_softmax))
    )


def optimize_curriculum(teacher, student, generator, epoch, batch_size = 16):
    batch = []
    weights = [0.] * 4 + list(np.linspace(0, 1., 46)) + [1.] * 200

    n_iter = batch_size
    for i in range(n_iter):
        encoding_size = 128
        specimens = np.random.uniform(-3.3, 3.3, size = (30, encoding_size))
        x = 0
        label = np.eye(10)[np.random.randint(10)]
        while x < 10:
            x += 1
            encodings = specimens
            with torch.no_grad():
                images = generator(torch.tensor(
                    specimens
                ).float().cuda())
                multipliers = [.2126, .7152, .0722]
                multipliers = np.expand_dims(multipliers, 0)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.expand_dims(multipliers, -1)
                multipliers = np.tile(multipliers, [1, 1, 32, 32])
                multipliers = torch.Tensor(multipliers).to(setup.device)
                images = images * multipliers
                images = images.sum(axis = 1, keepdims = True)

                teacher_predictions = teacher(images).detach().cpu()
                student_predictions = student(images).detach().cpu()
            losses = [
                curriculum_loss(np.array(s), np.array(i), label, weights[epoch])
                for s, i in zip(
                    teacher_predictions, student_predictions
                )
            ]
            indexes = np.argsort(losses)
            image = images[indexes[0]: indexes[0] + 1]
            specimens = specimens[indexes[:10]]
            specimens = np.concatenate([
                specimens,
                specimens + np.random.normal(scale = .5, size = (10, encoding_size)),
                specimens + np.random.normal(scale = .5, size = (10, encoding_size))
            ])
        batch.append(image)
    return torch.cat(batch, axis = 0)
