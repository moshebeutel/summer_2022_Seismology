import argparse
import logging
import random

import numpy as np
import torch

from snr.calc_snr import CalcSNR, str2snr_strategy
from snr.conversions import snr_to_factor
from utils.common import config_logger, set_seed

import os
from obspy import read


def create_noised_traces_shifted_noise(trace, full_noise_trace, factor, num_of_shifts: int = 5,
                                       sampling_rate: int = 100) -> list[torch.tensor]:
    from_seconds = lambda sh: sh * sampling_rate
    num_samples = trace.shape[-1]
    shifted_noise_traces = [
        trace.clone() + factor * full_noise_trace[from_seconds(sh):(num_samples + from_seconds(sh))].clone()
        for sh in range(num_of_shifts + 1)]

    return shifted_noise_traces


def get_random_noise_trace(noises_path: str = 'Noises'):
    noises_folders = os.listdir(noises_path)
    random_folder = random.choice(noises_folders)
    noises_in_folder = os.listdir(os.path.join(noises_path, random_folder))
    random_noise = random.choice(noises_in_folder)
    st = read(os.path.join(noises_path, random_folder, random_noise))
    return torch.from_numpy(st[0].data)


def get_random_noise_window(tr, desired_window_size=3001):
    full_length = tr.shape[0]
    if full_length <= desired_window_size:
        return full_length
    rand_idx = random.randint(0, full_length - desired_window_size)
    return tr[rand_idx:rand_idx + desired_window_size]


def create_noise_trace(whole_trace, onset_not_adjusted, noise_length, delta=10):
    assert noise_length < onset_not_adjusted, 'Noise trace must not get beyond onset'
    start = random.randint(0, onset_not_adjusted - noise_length - delta)
    return whole_trace[:, start:(start + noise_length)]


def create_noise_trace_batch(whole_trace, onset_not_adjusted, noise_length, desired_number_of_noise_traces, delta=10):
    onset_not_adjusted = int(onset_not_adjusted)
    number_of_noise_traces = min(onset_not_adjusted - noise_length - delta, desired_number_of_noise_traces)
    return [create_noise_trace(whole_trace, onset_not_adjusted, noise_length)
            for _ in range(number_of_noise_traces)]


def create_noisy_dataset(orig_sig_traces,
                         orig_whole_traces,
                         orig_onsets,
                         orig_onsets_not_adjusted,
                         SNRs,
                         desired_snr,
                         snr_startegy,
                         noisy_versions_for_each_trace=1):
    noisy_test_trace_list = []
    calc_snr_energy_ratio = CalcSNR(snr_startegy)
    logger = logging.getLogger()
    logger.info('create_noisy_dataset')
    for trace_idx in [0]:  # range(len(high_SNR_traces)):
        assert orig_sig_traces.dim() == 3, f'orig_sig_traces.dim() error, expected 3. Got {orig_sig_traces.dim()}'

        noisy_traces = create_noise_trace_batch(orig_whole_traces[trace_idx],
                                                orig_onsets_not_adjusted[trace_idx],
                                                orig_sig_traces.shape[2],
                                                noisy_versions_for_each_trace)

        trc = torch.clone(orig_sig_traces[trace_idx])
        original_trace_snr = SNRs[trace_idx]
        label = int(orig_onsets[trace_idx])
        for nt in noisy_traces:
            factor = snr_to_factor(trace=trc, label=label, clean_snr=original_trace_snr,
                                   desired_snr=desired_snr, noise_trace=nt,
                                   calc_snr=calc_snr_energy_ratio)
            noisy_test_trace_list.append(torch.clone(trc + (factor * nt)))

    return torch.stack(noisy_test_trace_list)


def main(args):
    batch = torch.load(args.base_dir + 'batch.pt')
    SNR = np.load(args.base_dir + f'SNR_{args.snr_calc_strategy}.npy')

    traces = batch['X']
    high_SNR_traces_indices = np.argwhere(SNR >= args.snr_threshold).squeeze()
    SNR_of_high_SNR_traces = SNR[high_SNR_traces_indices]
    high_SNR_traces = batch['X'][high_SNR_traces_indices]
    high_SNR_traces_labels = batch['onset_sample'][high_SNR_traces_indices]

    logger = logging.getLogger()
    logger.info(f'There are {traces.shape[0]} traces in batch')
    logger.debug(f'SNR values in batch {SNR} ')
    logger.info(f'There are {len(high_SNR_traces_indices)} samples with high SNRs')
    logger.info(f'high_SNR_traces shape {high_SNR_traces.shape}')
    logger.debug(f'Samples with high SNRs indices {high_SNR_traces_indices.T}')
    logger.debug(f'The high SNRs {SNR_of_high_SNR_traces.T}')

    desired_snr = 5
    noisy_versions_for_each_trace = 3
    noisy_dataset = \
        create_noisy_dataset(orig_sig_traces=high_SNR_traces,
                             orig_whole_traces=batch['whole_trace'][high_SNR_traces_indices],
                             orig_onsets=high_SNR_traces_labels,
                             orig_onsets_not_adjusted=batch['onset_sample_not_adjusted'][high_SNR_traces_indices],
                             SNRs=SNR_of_high_SNR_traces,
                             desired_snr=desired_snr,
                             snr_startegy=str2snr_strategy(args.snr_calc_strategy),
                             noisy_versions_for_each_trace=noisy_versions_for_each_trace)
    torch.save(noisy_dataset, args.datasets_dir +
               f'noisy_dataset_{noisy_dataset.shape[0]}_traces__snr_{desired_snr}_'
               f'_{noisy_versions_for_each_trace}_noisy_versions_for_each_original_trace.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Noisy Dataset")
    parser.add_argument("--base_dir", type=str,
                        default='/home/moshe/Documents/Helmholtz Summer 2022 Internship/Academic/code/ETHZ_phasenet/')
    parser.add_argument("--dataset", type=str, default='ethz',
                        choices=['ethz', 'geofon', 'stead', 'instance', 'iquique', 'lendb', 'neic', 'scdecs'])

    parser.add_argument("--datasets_dir", type=str,
                        default='/home/moshe/Documents/Helmholtz Summer 2022 Internship/'
                                'Academic/code/ETHZ_phasenet/datasets/')

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

    exp_name = f'Create Noisy Dataset -  Dataset_{args.dataset} Noise Added From {args.noise_source} ' \
               f'SNR Calculation Function {args.snr_calc_strategy}'
    logging.info(f'Experiment Name : {exp_name}')

    # Weights & Biases
    # if args.wandb:
    #     wandb.init(project="seisbench_uncertainty", entity="emg_diff_priv", name=exp_name)
    #     wandb.config.update(args)

    main(args)


