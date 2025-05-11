import os
import argparse

from torch.backends import cudnn
from utils.utils import *
import random
import torch
import numpy as np

from solver_OT import Solver
from utils.logger import get_logger
import optuna  # 新增导入

def str2bool(v):
    return v.lower() in ('true')

def objective(trial):
    # Set the parameter adjustment range
    config = argparse.Namespace(
        seed=1,
        mode='train',
        data_path='dataset/PSM',
        dataset='PSM',
        input_c=25,
        output_c=25,
        model_save_path='model_params',
        results='results',
        mask_ratio=trial.suggest_float('mask_ratio', 0.01, 0.1),
        k=trial.suggest_float('k', 1.0, 5.0),
        num_proto=trial.suggest_categorical('num_proto', [4, 8, 16]),
        len_map=trial.suggest_categorical('len_map', [4, 8, 16]),
        anomaly_ratio=0.5,
        win_size=trial.suggest_int('win_size', 50, 150),
        batch_size=64,
        lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        num_epochs=5,
        e_layers=3,
        d_model=512,
        n_heads=8,
    )

    solver = Solver(vars(config))
    solver.train()

    # Load validation score, e.g. validation loss for last epoch (smaller is better)
    vali_loss = solver.vali(solver.vali_loader)
    return -vali_loss  # Optuna is maximized by default, so it takes a negative value.

def main(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':

        if not os.path.exists(config.results):
            os.makedirs(config.results)
        logger = get_logger(config.results, __name__, str(config.dataset) + '_{}_{}_{}_{}_{}.log'
                             .format(config.k, config.num_proto, config.len_map, config.mask_ratio, config.anomaly_ratio))
        logger.info(config)

        solver.test(logger)

    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--optuna', action='store_true', help='Whether to use Optuna for automatic parameter adjustment')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='dataset/PSM')
    parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--input_c', type=int, default=25)
    parser.add_argument('--output_c', type=int, default=25)
    parser.add_argument('--model_save_path', type=str, default='model_params')
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--mask_ratio', type=float, default=0.05)
    parser.add_argument('--k', type=float, default=2)
    parser.add_argument('--num_proto', type=int, default=8)
    parser.add_argument('--len_map', type=int, default=8)
    parser.add_argument('--anomaly_ratio', type=float, default=0.5)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--use_spot', action='store_true', help='Whether to use the SPOT threshold')

    config = parser.parse_args()

    if config.optuna:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        print("\nBest Trial:")
        print(study.best_trial)
    else:
        args = vars(config)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        main(config)
