import os
import random

import torch
from obspy import read, Stream

from snr.calc_snr import CalcSNR, SnrCalcStrategy
from snr.conversions import snr_to_factor
from utils.common import get_residual, predict, try_get_saved_pt, standardize_trace
from tqdm import tqdm


def find_large_error_traces(dataset: torch.tensor, labels: torch.tensor, model: torch.nn.Module,
                            threshold_samples: int = 100, eval_fn=None) -> list[int]:
    assert dataset.shape[0] == labels.shape[0]
    large_error_idxs = [i for i in tqdm(range(dataset.shape[0])) if
                        get_residual(prediction=predict(trace=dataset[i], model=model, eval_fn=eval_fn),
                                     label=labels[i]) > threshold_samples]

    return large_error_idxs


def search_large_errors_given_desired_snr(model: torch.nn.Module, dataset: torch.tensor, labels: torch.tensor,
                                          noise_traces: torch.tensor,
                                          desired_snr: int = 10, calc_snr=CalcSNR(SnrCalcStrategy.ENERGY_RATIO)):
    noised_traces_list = create_noisy_traces(calc_snr=calc_snr, dataset=dataset, desired_snr=desired_snr, labels=labels,
                                             noise_traces=noise_traces)

    noisy_dataset = torch.stack(noised_traces_list, dim=0)

    large_error_indexes = find_large_error_traces(dataset=noisy_dataset, labels=labels, model=model)

    return large_error_indexes


def get_n_random_noises(num_noises: int,
                        desired_window_size: int, filename: str, noises_path: str, sampling_rate: float,
                        silent_exception_prints: bool = False,
                        force_resample: bool = False, save_to_pt: bool = False) -> torch.tensor:
    random_n_noises = None
    if not force_resample:
        random_n_noises = try_get_saved_pt(filename=filename, directory=noises_path)
    if random_n_noises is None:
        # Sample new ones
        random_noises_list = [get_random_noise_window(tr=get_random_noise_trace(noises_path=noises_path,
                                                                                sampling_rate=sampling_rate,
                                                                                silent_exception_prints=silent_exception_prints),
                                                      desired_window_size=desired_window_size)
                                       for _ in tqdm(range(num_noises))]
        print(f'Created a list of {int(len(random_noises_list))} random noises of shape {random_noises_list[0].shape}')
        print('stack to tensor')
        random_n_noises = torch.stack(random_noises_list, dim=0)
        print(f'Stacked to tensor of shape {random_n_noises.shape}')
        if save_to_pt:
            torch.save(random_n_noises, os.path.join(noises_path, filename))

    return random_n_noises


def create_noisy_traces(dataset, desired_snr, labels, noise_traces, calc_snr) -> (list[torch.tensor], list[int]):
    num_of_traces = dataset.shape[0]
    assert num_of_traces == noise_traces.shape[
        0], f'Expected one noise trace for each trace. Got {noise_traces.shape[0]} noise_traces for' \
            f' {num_of_traces} traces.'
    assert num_of_traces == labels.shape[
        0], f'Expected one label for each trace. Got {labels.shape[0]} labels for {num_of_traces} traces.'
    noised_traces_list = []
    insane_factor_4_snr_to_factor = 100000000
    not_included_indices = []
    factors = []
    for i in tqdm(range(num_of_traces)):
        # Calculate SNR before merging noise
        tr = dataset[i]
        label = int(labels[i])

        # Take the initial part of noise the size of dataset traces
        noise = noise_traces[i][:dataset.shape[-1]]
        # TODO enable torch tensor in CalcSNR
        # TODO debug  infinite factor searches
        clean_snr = calc_snr(tr[0].numpy(), label)
        # Find suitable factor to create the desired SNR
        factor = snr_to_factor(trace=tr.clone(), label=label,
                               noise_trace=noise,
                               clean_snr=clean_snr,
                               desired_snr=desired_snr,
                               calc_snr=calc_snr,
                               insane_factor=insane_factor_4_snr_to_factor)

        if factor < insane_factor_4_snr_to_factor:
            # Merge the trace with a multiplication of the noise
            # TODO Reduce noise mean
            noisy_trace = tr + factor * noise
            noised_traces_list.append(noisy_trace)
            factors.append(factor)
        else:
            not_included_indices.append(i)
    assert len(noised_traces_list) + len(not_included_indices) == num_of_traces, \
        f'Expected len(noised_traces_list) + len(not_included_indices) == num_of_traces.' \
        f' Got {len(noised_traces_list)} + {len(not_included_indices)} != {num_of_traces}'
    return noised_traces_list, factors, not_included_indices


def get_random_noise_window(tr, desired_window_size=3001):
    full_length = tr.shape[-1]
    if full_length <= desired_window_size:
        return full_length
    rand_idx = random.randint(0, full_length - desired_window_size)
    tr = tr[:, rand_idx:rand_idx + desired_window_size]
    # tr = standardize_trace(tr) if standardize else tr
    return tr


def trim_to_maximum_aligned_time_segment(st: Stream) -> Stream:
    assert len(st) == 3, f'Expected stream of 3 channels. Got {len(st)}'
    # Find maximum starttime and minimum endtime of traces to create a trimmed aligned stream
    max_start = max([tr.stats.starttime for tr in st.traces])
    min_end = min([tr.stats.endtime for tr in st.traces])
    st_trimmed = st.trim(max_start, min_end)
    return st_trimmed


def get_random_noise_trace(noises_path: str = 'Noises', sampling_rate: float = -1, silent_exception_prints=False)->torch.tensor:
    st = None
    noises_folders = os.listdir(noises_path)
    random_folder = random.choice(noises_folders)
    noises_in_folder = os.listdir(os.path.join(noises_path, random_folder))
    while st is None:
        random_noise = random.choice(noises_in_folder)
        try:
            st = read(os.path.join(noises_path, random_folder, random_noise))
        except (IOError, TypeError) as exception:
            if not silent_exception_prints:
                print(exception)

    st = trim_to_maximum_aligned_time_segment(st)
    st = st.normalize()
    st = st.interpolate(sampling_rate=sampling_rate) if sampling_rate > 0 else st
    # Deal with cases where after interpolation traces are not equal length
    min_num_samples = min([int(len(tr)) for tr in st.traces])
    return torch.stack([torch.from_numpy(tr.data[:min_num_samples]) for tr in st.traces], dim=0)
