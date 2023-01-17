import argparse
import logging
import torch
from tqdm import tqdm
from math import ceil
from dataset_creation.utils import create_noisy_traces
from evaluation.metrics import Metrics
from snr.calc_snr import CalcSNR, SnrCalcStrategy
from utils.common import config_logger, set_seed, shuffle_tensors, load_dataset


def model_performance_on_dataset(model, dataset, labels, metric_func):
    pred_dist = eval_batch(dataset, model)
    predictions = torch.argmax(pred_dist[:, 0], axis=1)
    assert labels.shape[0] == predictions.shape[0]
    return metric_func(labels, predictions)


def eval_mse(model, traces, labels, ignore_events_above_samples_threshold=0, batch_size=0):
    num_traces = traces.shape[0]
    num_batches = 0
    if batch_size == 0:
        batch_size = num_traces
        if num_traces > 64:
            raise ValueError(f'You supplied a big model - {num_traces} traces - specify batch size')
    else:
        num_batches = ceil(num_traces / batch_size)

    se = torch.zeros((1,))
    i = 0
    for _ in tqdm(range(num_batches)):
        batch_traces, batch_labels = traces[i:i + batch_size], labels[i:i + batch_size]
        pred_dist = eval_batch(batch=batch_traces, model=model)
        if ignore_events_above_samples_threshold > 0:
            for j, l in enumerate(batch_labels):
                pred_dist[j, :2, :(int(batch_labels[j]) - ignore_events_above_samples_threshold)] = 0
                pred_dist[j, :2, (int(batch_labels[j]) + ignore_events_above_samples_threshold):] = 0
        predictions = torch.argmax(pred_dist[:, 0], dim=-1)
        se += torch.square(predictions - batch_labels).sum()
        i += batch_size
    return torch.sqrt(se / num_traces).item()


def eval_noisy_vs_original(pretrained_model, dataset_labels, noisy_dataset, original_dataset, metrics_f=Metrics.mse):
    logger = logging.getLogger()

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


def find_large_error_traces(dataset: torch.tensor,
                            labels: torch.tensor, model: torch.nn.Module,
                            threshold_samples: int = 100,
                            ignore_errors_larger_than_samples=3000, eval_fn=None) -> list[int]:
    assert dataset.shape[0] == labels.shape[0]
    predictions = max_onset_pred(eval_fn(dataset, model=model))
    residuals = torch.abs(predictions - labels)

    large_error_idxs = [i for i in tqdm(range(dataset.shape[0])) if
                        threshold_samples < residuals[i] < ignore_errors_larger_than_samples]

    return large_error_idxs


def search_large_errors_given_desired_snr(model: torch.nn.Module, dataset: torch.tensor, labels: torch.tensor,
                                          noise_traces: torch.tensor,
                                          desired_snr: int = 10, calc_snr=CalcSNR(SnrCalcStrategy.ENERGY_RATIO)):
    noised_traces_list = create_noisy_traces(calc_snr=calc_snr, dataset=dataset, desired_snr=desired_snr, labels=labels,
                                             noise_traces=noise_traces)

    noisy_dataset = torch.stack(noised_traces_list, dim=0)

    large_error_indexes = find_large_error_traces(dataset=noisy_dataset, labels=labels, model=model)

    return large_error_indexes


def eval_batch(batch, model):
    with torch.no_grad():
        pred = model(batch)
        pred = pred.cpu()
    return pred


def max_onset_pred(pred_probs):
    return torch.argmax(pred_probs[:, 0], dim=1)


def predict(trace, model, eval_fn=None):
    evaluation_fn = eval_batch if eval_fn is None else eval_fn
    return max_onset_pred(evaluation_fn(trace.unsqueeze(dim=0), model=model))


def get_residual(prediction: int, label: int) -> int:
    return int(torch.abs(prediction - label))
