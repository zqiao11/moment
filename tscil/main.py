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
    parser.add_argument('--agent', dest='agent', default='L2P', type=str,
                        choices=['SFT', 'Offline', 'L2P'],
                        help='Continual learning agent')

    # Ignore this arg for now
    parser.add_argument('--scenario', type=str, default='class',
                        choices=['class', 'domain'],
                        help='Scenario of the task steam. Current codes only include class-il')

    parser.add_argument('--stream_split', type=str, default='exp',
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
    parser.add_argument('--use_prototype', dest='use_prototype', type=boolean_string, default=True,
                        help='Use class prototype for classification.')
    parser.add_argument('--freeze_old_cls_weights', default=False, type=bool, )

    # General params
    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='Number of runs')
    parser.add_argument('--epochs', dest='epochs', default=1, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--lradj', type=str, default='constant')
    parser.add_argument('--early_stop', type=boolean_string, default=True)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--weight_decay', dest='weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--z_score_norm', type=boolean_string, default=False, help='conducting offline z score normalization')
    parser.add_argument('--prop', type=float, default=1.0,
                        help='Proportion of train data for each class in per task')
    
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
    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--pool_size', default=10, type=int,)
    parser.add_argument('--prompt_length', default=5,type=int, )
    parser.add_argument('--top_k', default=5, type=int, )
    parser.add_argument('--prompt_key', default=True, type=bool,)
    parser.add_argument('--prompt_key_init', default='uniform', type=str)
    parser.add_argument('--use_prompt_mask', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)

    # USed in engine.py
    parser.add_argument('--shared_prompt_pool', default=False, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.1, type=float)


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
