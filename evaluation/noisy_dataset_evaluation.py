import argparse
import logging
import os

import numpy as np
import torch

from evaluation.metrics import Metrics
from utils.common import config_logger, set_seed, shuffle_tensors, eval_batch, load_dataset


def model_performance_on_dataset(model, dataset, labels, metric_func):
    pred_dist = eval_batch(dataset, model)
    predictions = torch.argmax(pred_dist[:, 0], axis=1)
    assert labels.shape[0] == predictions.shape[0]
    return metric_func(labels, predictions)


def eval_noisy_vs_original(pretrained_model, dataset_labels, noisy_dataset, original_dataset, metrics_f=Metrics.mse):
    logger = logging.getLogger()

    logger.info('Shuffling traces')
    [noisy_dataset, original_dataset, dataset_labels] = \
        shuffle_tensors([noisy_dataset, original_dataset, dataset_labels])

    logger.info('Evaluate Noisy dataset')
    noisy_dataset_metric_result = model_performance_on_dataset(pretrained_model, noisy_dataset, dataset_labels,
                                                               metrics_f)
    logger.info(f'Noisy dataset metrics {metrics_f.__name__}: {noisy_dataset_metric_result}')
    logger.info('Evaluate original dataset')
    original_dataset_metric_result = model_performance_on_dataset(pretrained_model, original_dataset, dataset_labels,
                                                                  metrics_f)
    logger.info(f'Original dataset metrics {metrics_f.__name__}: {original_dataset_metric_result}')
    logger.info(f'Metrics difference {noisy_dataset_metric_result - original_dataset_metric_result}')

    return noisy_dataset_metric_result, original_dataset_metric_result


def eval_model_dataset_multi_metric(pretrained_model, dataset_labels, dataset, metrics_list):
    result_list = []
    for metric_f in metrics_list:
        result = eval_model_dataset(pretrained_model, dataset_labels, dataset, metrics_f=metric_f)
        result_list.append(result)
    return result_list


def eval_model_dataset(pretrained_model, dataset_labels, dataset, metrics_f=Metrics.mse):
    logger = logging.getLogger()

    logger.info('Shuffling traces')
    [dataset, dataset_labels] = \
        shuffle_tensors([dataset, dataset_labels])

    logger.info('Evaluate dataset')
    dataset_metric_result = model_performance_on_dataset(pretrained_model, dataset, dataset_labels,
                                                         metrics_f)
    logger.info(f'Dataset metrics {metrics_f.__name__}: {dataset_metric_result}')

    return dataset_metric_result


def main(args):
    logger = logging.getLogger()
    noisy_dataset = load_dataset(args.datasets_dir + args.noisy_dataset_filename)
    original_dataset = load_dataset(args.datasets_dir + args.original_dataset_filename)
    dataset_labels = load_dataset(args.datasets_dir + args.dataset_labels)

    assert noisy_dataset.shape[0] == original_dataset.shape[0], \
        f'Loaded datasets with different sizes, {noisy_dataset.shape[0]} and {original_dataset.shape[0]}'
    assert noisy_dataset.shape[0] == dataset_labels.shape[0], \
        f'Loaded labels with different size, {dataset_labels.shape[0]} instead of {noisy_dataset.shape[0]}'

    logger.info(f'There are {noisy_dataset.shape[0]} traces to evaluate')
    logger.info(f'Load pretrained model {args.pretrained_model_filename}')
    pretrained_model = torch.load(args.pretrained_model_dir + args.pretrained_model_filename)
    eval_noisy_vs_original(pretrained_model, dataset_labels, noisy_dataset, original_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Noisy Dataset Evaluation")
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
