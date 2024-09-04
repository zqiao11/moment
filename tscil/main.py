# -*- coding: UTF-8 -*-
import argparse
import sys
import os
import torch
import time
from utils.utils import Logger, boolean_string
from experiment import experiment_multiple_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the continual learning agent on task sequence')

    # #################### Main setting for the experiment ####################
    parser.add_argument('--agent', dest='agent', default='SFT', type=str,
                        choices=['SFT', 'Offline', 'L2P'],
                        help='Continual learning agent')

    # Ignore this arg for now
    parser.add_argument('--scenario', type=str, default='class',
                        choices=['class', 'domain'],
                        help='Scenario of the task steam. Current codes only include class-il')

    parser.add_argument('--stream_split', type=str, default='all',
                        choices=['val', 'exp', 'all'],
                        help='The split of the tasks stream: val tasks, exp tasks or all the tasks')

    parser.add_argument('--data', dest='data', default='uwave', type=str,
                        choices=['har', 'uwave', 'dailysports', 'grabmyo', 'wisdm',
                                 'ninapro', 'sines'])

    # Moment
    parser.add_argument('--freeze_encoder', dest='freeze_encoder',
                        type=boolean_string, default=True,)
    parser.add_argument('--freeze_embedder', dest='freeze_embedder',
                        type=boolean_string, default=True, )
    parser.add_argument('--reduction', type=str, default='concat',
                        choices=['mean', 'concat'],)

    # Classifier. Ignore these args for now.
    parser.add_argument('--head', dest='head', default='Linear', type=str,
                        choices=['Linear', 'CosineLinear', 'SplitCosineLinear'])
    parser.add_argument('--criterion', dest='criterion', default='CE', type=str,
                        choices=['CE', 'BCE'])  # Main classification loss and activation of head
    parser.add_argument('--ncm_classifier', dest='ncm_classifier', type=boolean_string, default=False,
                        help='Use NCM classifier or not. Only work for ER-based methods.')

    # General params
    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='Number of runs')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--lradj', type=str, default='constant')
    parser.add_argument('--early_stop', type=boolean_string, default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--z_score_norm', type=boolean_string, default=False, help='conducting offline z score normalization')

    # #################### Nuisance variables  ####################
    parser.add_argument('--tune', type=boolean_string, default=False, help='flag of tuning')
    parser.add_argument('--debug', type=boolean_string, default=True,  help='flag of debugging') # will save the results in a 'debug' folder
    parser.add_argument('--seed', dest='seed', default=1234, type=int)
    parser.add_argument('--device', dest='device', default='cuda', type=str)
    parser.add_argument('--verbose', type=boolean_string, default=True)
    parser.add_argument('--exp_start_time', dest='exp_start_time', type=str)
    parser.add_argument('--fix_order', type=boolean_string, default=False,
                        help='Fix the class order for different runs')
    parser.add_argument('--cf_matrix', type=boolean_string, default=False,
                        help='Plot confusion matrix or not')
    parser.add_argument('--tsne', type=boolean_string, default=False,
                        help='Visualize the feature space of learner with TSNE')
    parser.add_argument('--tsne_g', type=boolean_string, default=False,
                        help='Visualize the feature space of generator with TSNE')

    # ######################## Methods-related params ###########################
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set directories
    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_0 = './result/exp/' if not args.debug else './result/exp/debug'
    exp_path_1 = 'Moment-Large' + '_' + args.data
    exp_path_2 = args.agent + '_' + exp_start_time
    exp_path = os.path.join(exp_path_0, exp_path_1, exp_path_2)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    log_dir = exp_path + '/' + 'log.txt'
    sys.stdout = Logger(log_dir)
    args.exp_path = exp_path  # One experiment with multiple runs
    print(args)

    experiment_multiple_runs(args)
