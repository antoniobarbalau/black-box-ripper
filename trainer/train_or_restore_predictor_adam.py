import numpy as np
import os
import setup
import torch

def train_or_restore_predictor_adam(
    model, dataset,
    loss_type = 'categorical',
    n_epochs = 50,
):
    model_exists = False
    ckpt_path = f'./checkpoints/{model.name}_state_dict'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location = setup.device))
        model.to(setup.device)
        model_exists = True

    training_was_in_progress = False
    root_optimizer_ckpt_path = f'optimizer_for_{model.name}_state_dict'
    optimizer_ckpt_path = root_optimizer_ckpt_path
    for filename in os.listdir('./checkpoints'):
        if optimizer_ckpt_path in filename:
            training_was_in_progress = True
            optimizer_ckpt_path = filename

    if model_exists and not training_was_in_progress:
        model.eval()
        return

    lr = 0.001
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr,
    )
    if training_was_in_progress:
        optimizer.load_state_dict(torch.load(f'./checkpoints/{optimizer_ckpt_path}'))
    loss_function = torch.nn.CrossEntropyLoss()
    if loss_type == 'binary':
        loss_function = torch.nn.BCELoss()

    has_val = 'val_dataset' in dataset.__dict__
    best_acc = 0.

    starting_epoch_n = 0
    if training_was_in_progress:
        starting_epoch_n = int(optimizer_ckpt_path.split('_')[-1])
    for epoch in range(starting_epoch_n + 1, n_epochs + 1):
        model.train(True)
        dataloader = dataset.train_dataloader(epoch)

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
            print(f'{epoch}, {iter_n}, {acc}', end = '\r')

            loss.backward()
            optimizer.step()

        model.eval()
        dataloader = (
            dataset.val_dataloader()
            if has_val and 'student' not in model.name else
            dataset.test_dataloader()
        )
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
        accs /= n_samples
        print(f'{epoch}, {accs}                        ')
        if not has_val or 'student' in model.name:
            torch.save(model.state_dict(), ckpt_path)
        elif accs > best_acc:
            best_acc = accs
            torch.save(model.state_dict(), ckpt_path)
        new_checkpoint_path = f'{root_optimizer_ckpt_path}_{epoch}'
        torch.save(optimizer.state_dict(), f'./checkpoints/{new_checkpoint_path}')
        if os.path.exists(f'./checkpoints/{optimizer_ckpt_path}'):
            os.unlink(f'./checkpoints/{optimizer_ckpt_path}')
        optimizer_ckpt_path = new_checkpoint_path
    # os.unlink(f'./checkpoints/{optimizer_ckpt_path}')
