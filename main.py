import os
import pathlib
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchsummary
from apex import amp
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console, step_flagging
from utils.plotter import plot_results
from utils.ema import EMA
from utils.model import Model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def adjust_lr_beta1(optimizer, lr, beta1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, 0.999)  # Only change beta1


def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def main(cfg, device):
    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    # logging ----------------------------------------------------------------------------------------------------------------------------------------
    logger_root = f'Results/{cfg.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.resume is None:
        result_dir = os.path.join(logger_root, f'{cfg.log}-{logtime}')
        logger = Logger(logging_dir=result_dir, DEBUG=False)
        logger.set_logfile(logfile_name='log.txt')
    else:
        result_dir = os.pardir.split(cfg.resume)[0]  # TODO
        logger = Logger(logging_dir=result_dir, DEBUG=False)
        logger.set_logfile('resumed-log.txt')
    save_config(cfg, f'{result_dir}/config.cfg')
    save_params(cfg, f'{result_dir}/params.json', json_format=True)
    logger.debug(f'Result Path: {result_dir}')

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    n_classes = cfg.n_classes
    net = Model(arch=cfg.net, num_classes=n_classes, pretrained=True)
    net_ema = Model(arch=cfg.net, num_classes=n_classes, pretrained=True)

    optimizer = build_sgd_optimizer(net.parameters(), cfg.lr, cfg.weight_decay)
    scheduler = build_cosine_lr_scheduler(optimizer, cfg.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=round(cfg.epochs/3), gamma=0.1)
    opt_lvl = 'O1' if cfg.use_fp16 else 'O0'
    [net, net_ema], optimizer = amp.initialize([net.to(device), net_ema.to(device)], optimizer, opt_level=opt_lvl,
                                               keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)
    # log network
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())
    # input_size = (3, cfg.crop_size, cfg.crop_size)
    # torchsummary.summary(net, input_size=input_size, batch_size=cfg.batch_size)

    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    transform = build_transform(rescale_size=cfg.rescale_size, crop_size=cfg.crop_size)
    if cfg.dataset == 'food101n':
        dataset = build_food101n_dataset(os.path.join(cfg.database, cfg.dataset), CLDataTransform(transform['train']), transform['test'])
    elif cfg.dataset == 'clothing1m':
        dataset = build_clothing1m_dataset(os.path.join(cfg.database, cfg.dataset), CLDataTransform(transform['train']), transform['test'])
    else:
        raise AssertionError(f'{cfg.dataset} is not supported yet!')
    train_loader = DataLoader(dataset['train'], batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()

    # resume -----------------------------------------------------------------------------------------------------------------------------------------
    if cfg.resume is not None:
        assert os.path.isfile(cfg.resume), 'no checkpoint.pth exists!'
        logger.debug(f'---> loading {cfg.resume} <---')
        checkpoint = torch.load(f'{cfg.resume}')
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        best_epoch = checkpoint['best_epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best_accuracy = 0.0
        best_epoch = None
    scheduler.last_epoch = start_epoch

    ema = EMA(net, alpha=0.99)
    ema.apply_shadow(net_ema)

    flag = 0
    tau_c_max = 0.95
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()

        # pre-step in this epoch
        net.train()
        train_loss.reset()
        train_accuracy.reset()
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        logger.debug(f'Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  Lr:[{curr_lr[0]:.5f}]')

        if epoch < cfg.warmup_epochs:
            threshold_clean = min(cfg.tau_clean * epoch / cfg.warmup_epochs, cfg.tau_clean)
        else:
            threshold_clean = (tau_c_max - cfg.tau_clean) * (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs) + cfg.tau_clean
        print_to_console(f'> threshold_clean: {threshold_clean:.5f}', color='red')
        # train this epoch
        for it, sample in enumerate(train_loader):
            s = time.time()
            optimizer.zero_grad()
            # indices = sample['index']   # 'torch.Tensor'
            x1, x2 = sample['data']
            x1, x2 = x1.to(device), x2.to(device)
            y = sample['label'].to(device)

            logits1 = net(x1)
            logits2 = net(x2)
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)

            N, C = logits1.shape
            given_labels = torch.full(size=(N, C), fill_value=cfg.eps/(C - 1)).to(device)
            given_labels.scatter_(dim=1, index=torch.unsqueeze(y, dim=1), value=1-cfg.eps)
            with torch.no_grad():
                logits1_ema = net_ema(x1)
                logits2_ema = net_ema(x2)
                soft_labels = (F.softmax(logits1_ema, dim=1) + F.softmax(logits2_ema, dim=1)) / 2

                prob_clean = 1 - js_div(probs1, given_labels)

            if epoch < cfg.warmup_epochs:
                if flag == 0:
                    step_flagging(f'start the warm-up step for {cfg.warmup_epochs} epochs.')
                    flag += 1
                losses = cross_entropy(logits1, given_labels, reduction='none') + cross_entropy(logits2, given_labels, reduction='none')
                loss = losses[prob_clean >= threshold_clean].mean()
            else:
                if flag == 1:
                    step_flagging('start the robust learning step.')
                    flag += 1

                target_labels = given_labels.clone()
                # clean samples
                idx_clean = (prob_clean >= threshold_clean).nonzero(as_tuple=False).squeeze(dim=1)
                _, preds1 = probs1.topk(1, 1, True, True)
                _, preds2 = probs2.topk(1, 1, True, True)
                disagree = (preds1 != preds2).squeeze(dim=1)
                agree = (preds1 == preds2).squeeze(dim=1)
                unclean = (prob_clean < threshold_clean)
                idx_ood = (disagree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
                idx_id = (agree * unclean).nonzero(as_tuple=False).squeeze(dim=1)
                target_labels[idx_id] = soft_labels[idx_id]
                target_labels[idx_ood] = F.softmax(soft_labels[idx_ood] / 10, dim=1)

                # classification loss
                losses = cross_entropy(logits1, target_labels, reduction='none') + cross_entropy(logits2, target_labels, reduction='none')
                loss_c = losses.mean()

                # consistency loss
                sign = torch.ones(N).to(device)
                sign[idx_ood] *= -1
                losses_o = symmetric_kl_div(probs1, probs2) * sign
                loss_o = losses_o.mean()

                # final loss
                loss = (1 - cfg.alpha) * loss_c + loss_o * cfg.alpha

            train_acc = accuracy(logits1, y, topk=(1,))
            train_accuracy.update(train_acc[0], x1.size(0))
            train_loss.update(loss.item(), x1.size(0))
            if cfg.use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            ema.update_params(net)
            ema.apply_shadow(net_ema)

            epoch_train_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        # evaluate this epoch
        test_accuracy = evaluate(test_loader, net, device)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')

        # post-step in this epoch
        scheduler.step()

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, filename=f'{result_dir}/checkpoint.pth')

        # logging this epoch
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt')

    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--log_prefix', type=str)
    parser.add_argument('--log_freq', type=int)
    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    if config.dataset.startswith('cifar'):
        config.log = f'{config.net}-{config.noise_type}_closeset{config.closeset_ratio}_openset{config.openset_ratio}-{config.log_prefix}'
    else:
        config.log = f'{config.net}-{config.log_prefix}'
    print(config)
    return config


if __name__ == '__main__':
    params = parse_args()
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
