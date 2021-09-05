import os
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, choices=['leduc_holdem',
                                               'texas_holdem',
                                               'connect_four',
                                               ],
                    help='env_id')
parser.add_argument('-training_setting', type=str, choices=['single_policy',
                                                             'simple_ensemble',
                                                             'implicit_ensemble'],
                    help='train setting')
parser.add_argument('-logdir', type=str,
                    help='logging directory')
parser.add_argument('--main_ckpt', type=str,
                    help='main checkpoint path to restore')
parser.add_argument('--test_ckpt', type=str,
                    help='main checkpoint path to restore')
parser.add_argument('--no_render', action='store_true',
                    help='main checkpoint path to restore')
parser.add_argument('--num_cpus', type=int, default=6,
                    help='number of cpu cores to run the code')
parser.add_argument('--train_step_multiplier', type=int, default=1500,
                    help='parameter determining the total steps for training')

def process_cmd_args(args):
    args.env = 'pettingzoo.classic.' + args.env + '_v1'
    args.env = importlib.import_module(args.env)
    if args.main_ckpt and args.main_ckpt.startswith('~/'):
        args.main_ckpt = os.path.expanduser(args.main_ckpt)
    if args.test_ckpt and args.test_ckpt.startswith('~/'):
        args.test_ckpt = os.path.expanduser(args.test_ckpt)
    return args

cmd_args = parser.parse_args()
cmd_args = process_cmd_args(cmd_args)