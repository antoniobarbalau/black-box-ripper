import cv2
import numpy as np
import torch

def plot_samples(samples, save = None):
    samples = samples.detach().cpu()
    samples = samples.permute(0, 2, 3, 1).clamp(-1., 1.)
    samples = samples * .5 + .5
    samples = samples.numpy()[:100]
    samples = samples.reshape(10, 10, samples.shape[1], samples.shape[2], 3)
    samples = np.concatenate(samples, axis = 1)
    samples = np.concatenate(samples, axis = 1)
    samples = np.uint8(samples * 255.)

    # cv2.imshow(f'generated_samples', samples)
    # cv2.waitKey(1)

    if save is not None:
        cv2.imwrite(f'./split_generated/{save}.png', samples)

