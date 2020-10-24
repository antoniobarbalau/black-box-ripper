import numpy as np
import os
import torch
import setup

def train_or_restore_predictor(
    model, dataset,
    loss_type = 'categorical',
    n_epochs = 50,
):
    ckpt_path = f'./checkpoints/{model.name}_state_dict'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=setup.device))
        model.to(setup.device)
        model.eval()
        return

    def set_lr(optimizer, lr):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-grad_clip, grad_clip)

    lr = 0.01
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = lr,
        momentum = .9,
        weight_decay = 5e-4
    )
    learning_rate_decay_start = 17
    learning_rate_decay_every = 1
    learning_rate_decay_rate = .9
    loss_function = torch.nn.CrossEntropyLoss()
    if loss_type == 'binary':
        loss_function = torch.nn.BCELoss()

    for epoch in range(n_epochs):
        model.train(True)
        dataloader = dataset.train_dataloader()

        if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            set_lr(optimizer, current_lr)
        else:
            current_lr = lr

        for iter_n, batch in enumerate(dataloader):
            images = batch[0].to(setup.device)
            targets = batch[1].to(setup.device)

            model.zero_grad()
            outputs = model(images)
            if loss_type == 'binary':
                outputs = torch.softmax(outputs, dim = -1)
            loss = loss_function(outputs, targets)
            if loss_type == 'binary':
                acc = outputs.max(1)[1].eq(targets.max(1)[1])
            else:
                acc = outputs.max(1)[1].eq(targets)
            acc = acc.float().mean().detach().cpu()

            if (iter_n) % 25 == 0:
                print(f'Epoch: {epoch}/{n_epochs}, {iter_n}, Accuracy: {acc}, Loss: {loss}')

            loss.backward()
            clip_gradient(optimizer, 0.1)
            optimizer.step()

        model.eval()
        dataloader = dataset.test_dataloader()
        accs = 0
        n_samples = 0
        for iter_n, batch in enumerate(dataloader):
            images = batch[0].to(setup.device)
            targets = batch[1].to(setup.device)
            n_samples += targets.shape[0]

            with torch.no_grad():
                outputs = model(images)
                acc = outputs.max(1)[1].eq(targets).float().sum()
                acc = acc.detach().cpu()
            accs += acc
        print(f'{model.name} accuracy: {accs / n_samples}')

        torch.save(model.state_dict(), ckpt_path)
