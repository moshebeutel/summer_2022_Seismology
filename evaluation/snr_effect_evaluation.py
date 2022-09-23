import argparse
import logging

import torch

from evaluation.metrics import Metrics
from evaluation.noisy_dataset_evaluation import eval_noisy_vs_original, eval_model_dataset, \
    eval_model_dataset_multi_metric
from snr.calc_snr import CalcSNR, SnrCalcStrategy
from utils.common import config_logger, set_seed, load_dataset
import pandas as pd


def main(args):
    logger = logging.getLogger()

    logger.info(f'Load pretrained model {args.pretrained_model_filename}')
    pretrained_model = torch.load(args.pretrained_model_dir + args.pretrained_model_filename)
    logger.info(f'Load original dataset {args.original_dataset_filename}')
    original_dataset = load_dataset(args.datasets_dir + args.original_dataset_filename)
    logger.info(f'Load dataset labels {args.dataset_labels}')
    dataset_labels = load_dataset(args.datasets_dir + args.dataset_labels)
    metric_functions = [Metrics.mae, Metrics.mse, Metrics.med_abs_err, Metrics.max_error]
    columns = ['snr'] + [f.__name__ for f in metric_functions]
    results = eval_model_dataset_multi_metric(pretrained_model, dataset_labels, original_dataset, metric_functions)
    data = [20] + results
    print(columns)
    print(data)
    df = pd.DataFrame(data=[data], columns=columns)
    for snr in range(1, 11):
        noisy_dataset_filename = f'ethz_phasenet_noisy_test_set_snr_{snr}.pt'
        logger.info(f'Load noisy dataset {noisy_dataset_filename}')
        noisy_dataset = load_dataset(args.datasets_dir + noisy_dataset_filename)
        assert noisy_dataset.shape[0] == original_dataset.shape[0], \
            f'Loaded datasets with different sizes, {noisy_dataset.shape[0]} and {original_dataset.shape[0]}'
        assert noisy_dataset.shape[0] == dataset_labels.shape[0], \
            f'Loaded labels with different size, {dataset_labels.shape[0]} instead of {noisy_dataset.shape[0]}'

        results = eval_model_dataset_multi_metric(pretrained_model, dataset_labels, noisy_dataset, metric_functions)
        data = [snr] + results
        df = pd.concat([df, pd.DataFrame(data=[data], columns=columns)])

    df.to_csv('results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SNR Effect Evaluation")
    parser.add_argument("--base_dir", type=str,
                        default='/home/moshe/Documents/Helmholtz Summer 2022 Internship/Academic/code/ETHZ_phasenet/')
    parser.add_argument("--dataset", type=str, default='ethz',
                        choices=['ethz', 'geofon', 'stead', 'instance', 'iquique', 'lendb', 'neic', 'scdecs'])

    parser.add_argument("--datasets_dir", type=str,
                        default='/home/moshe/Documents/Helmholtz Summer 2022 Internship/'
                                'Academic/code/ETHZ_phasenet/datasets/')
    parser.add_argument("--noisy_dataset_filename", type=str,
                        default='ethz_phasenet_noisy_test_set_snr_3.pt')
    parser.add_argument("--original_dataset_filename", type=str,
                        default='ethz_phasenet_original_test_set.pt')
    parser.add_argument("--dataset_labels", type=str,
                        default='ethz_phasenet_label_test_set.pt')
    parser.add_argument("--pretrained_model_dir", type=str,
                        default='/home/moshe/Documents/Helmholtz Summer 2022 Internship/'
                                'Academic/code/ETHZ_phasenet/models/')
    parser.add_argument("--pretrained_model_filename", type=str,
                        default='ethz_phasenet_model.pth')

    parser.add_argument("--seed", type=int, default=42, help="predefined seed value for reproducibility")

    parser.add_argument("--noise_source", type=str, default='trace_start',
                        choices=['dataset_start', 'trace_start'])
    parser.add_argument("--snr_calc_strategy", type=str, default='energy_ratio',
                        choices=['energy_ratio', 'max_amplitude_vs_rms_ratio', 'bandpass_filter'])
    parser.add_argument("--snr_threshold", type=int, default=10, help="dB value. Seek for SNRs above this threshold")

    args = parser.parse_args()

    log_level = logging.INFO

    config_logger(level=log_level)
    logging.getLogger().setLevel(log_level)
    logging.info(f'Logger set. Log level  = {logging.getLevelName(logging.getLogger().getEffectiveLevel())}')
    set_seed(args.seed)

    exp_name = f'Noisy Dataset evaluation -  Dataset_{args.dataset} Noise Added From {args.noise_source} ' \
               f'SNR Calculation Function {args.snr_calc_strategy}'
    logging.info(f'Experiment Name : {exp_name}')

    # Weights & Biases
    # if args.wandb:
    #     wandb.init(project="seisbench_uncertainty", entity="emg_diff_priv", name=exp_name)
    #     wandb.config.update(args)

    main(args)
