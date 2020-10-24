## Black-Box Ripper: Copying black-box models using generative evolutionary algorithms - NIPS 2020 - Official Implementation

### Setup
Install requirements

``` pip install -r requirements.txt```

Download pretrained models using:
 ```bash download_checkpoints.sh```


### Table 1 experiments - Experiments having CIFAR10 as true dataset

```python base_experiment.py --true_dataset cifar10 --teacher alexnet --student half_alexnet --generator cifar_100_6_classes_gan```

```python base_experiment.py --true_dataset cifar10 --teacher alexnet --student half_alexnet --generator cifar_100_10_classes_gan```

```python base_experiment.py --true_dataset cifar10 --teacher alexnet --student half_alexnet --generator cifar_100_40_classes_gan```

```python base_experiment.py --true_dataset cifar10 --teacher alexnet --student half_alexnet --generator cifar_100_90_classes_gan```

### Table 2 experiments - Experiments having Fashion-MNIST as true dataset

```python base_experiment.py --true_dataset split_fmnist --teacher lenet --student half_lenet --generator cifar_10_gan```

```python base_experiment.py --true_dataset split_fmnist --teacher lenet --student half_lenet --generator cifar_10_vae```

```python base_experiment.py --true_dataset fmnist --teacher vgg --student vgg --generator cifar_10_gan --optim sgd --epochs 50```

python base_experiment.py --true_dataset fmnist --teacher vgg --student vgg --generator cifar_10_vae --optim sgd --epochs 50


### Table 3 Experiments - Monkey Species as true dataset

The 10 Monkey Species dataset for the teacher is found at:

```https://www.kaggle.com/slothkong/10-monkey-species```

For CelebA-HQ as proxy, we used the PGAN from torch.hub:

```
def celeba_gan():
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
        'PGAN',
        model_name='celebAHQ-512',
        pretrained = True,
        useGPU = True
    )
    return model
```

For ImageNet-Cats-and-Dogs, we used the SNGAN-Projection found at:

```https://github.com/pfnet-research/sngan_projection```
