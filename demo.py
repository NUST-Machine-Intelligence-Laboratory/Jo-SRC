import os
import pathlib
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from torch.utils.data import DataLoader
from utils.core import evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console, step_flagging
from utils.model import Model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nclasses', type=int, required=True)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()

    init_seeds()
    device = set_device(args.gpu)

    transform = build_transform(rescale_size=224, crop_size=224)
    if args.dataset == 'food101n':
        dataset = build_food101n_dataset(os.path.join('Datasets', args.dataset), transform['train'], transform['test'])
        net = Model(args.arch, args.nclasses, 1).to(device)
    elif args.dataset == 'clothing1m':
        dataset = build_clothing1m_dataset(os.path.join('Datasets', args.dataset), transform['train'], transform['test'])
        net = Model(args.arch, args.nclasses, 2).to(device)
    else:
        raise AssertionError(f'{args.dataset} is not supported yet!')
    net.load_state_dict(torch.load(args.model_path))
    test_loader = DataLoader(dataset['test'], batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    test_accuracy = evaluate(test_loader, net, device)
    
    print(f'Test accuracy: {test_accuracy:.3f}')

