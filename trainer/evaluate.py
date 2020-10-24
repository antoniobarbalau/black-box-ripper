import numpy as np
import torch
import setup

def evaluate(model, dataset):
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
