import os
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.noisy_cifar import NoisyCIFAR10, NoisyCIFAR100
from data.food101 import Food101
from data.food101n import Food101N
from data.clothing1m import Clothing1M


# dataset --------------------------------------------------------------------------------------------------------------------------------------------
def build_transform(rescale_size=512, crop_size=448, s=1):
    cifar_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(size=crop_size),
        # RandAugment(),
        # ImageNetPolicy(),
        # Cutout(size=crop_size // 16),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=rescale_size),
        torchvision.transforms.CenterCrop(size=crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return {'train': train_transform, 'test': test_transform,
            'cifar_train': cifar_train_transform, 'cifar_test': cifar_test_transform}


def build_cifar100n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True)
    return {'train': train_data, 'test': test_data}


def build_food101n_dataset(root, train_transform, test_transform):
    train_data = Food101N(root, transform=train_transform)
    test_data = Food101(os.path.join(root, 'food-101'), split='test', transform=test_transform)
    return {'train': train_data, 'test': test_data}


def build_clothing1m_dataset(root, train_transform, test_transform):
    train_data = Clothing1M(root, split='train', transform=train_transform)
    valid_data = Clothing1M(root, split='val', transform=test_transform)
    test_data = Clothing1M(root, split='test', transform=test_transform)
    return {'train': train_data, 'test': test_data, 'val': valid_data}


# optimizer, scheduler -------------------------------------------------------------------------------------------------------------------------------
def build_sgd_optimizer(params, lr, weight_decay):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


def build_adam_optimizer(params, lr):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999))


def build_cosine_lr_scheduler(optimizer, total_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

