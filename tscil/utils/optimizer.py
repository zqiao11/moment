# -*- coding: UTF-8 -*-
from collections import defaultdict
import torch.optim as optim
from torch.optim import Adam


def set_optimizer(model, args, task_now=0):
    params = [{'params': model.parameters()}]
    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


# https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/utils/tools.py
def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=False):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.lr}
    elif args.lradj == 'step10':
        lr_adjust = {epoch: args.lr if epoch < 10 else args.lr * 0.1}
    elif args.lradj == 'step15':
        lr_adjust = {epoch: args.lr if epoch < 15 else args.lr * 0.1}
    elif args.lradj == 'step25':
        lr_adjust = {epoch: args.lr if epoch < 25 else args.lr * 0.1}
    elif args.lradj == 'step5':
        lr_adjust = {epoch: args.lr if epoch < 5 else args.lr * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))
