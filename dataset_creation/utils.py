import os
import random

import numpy as np
import pandas as pd

import torch
from obspy import read, Stream
from seisbench import generate as sbg
from seisbench.data import WaveformDataset
from torch.utils.data import DataLoader

from snr.conversions import snr_to_factor
from utils.common import try_get_saved_pt, sublist_complement, standardize_trace
from tqdm import tqdm


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
        # random_n_noises = standardize_trace(random_n_noises)
        # print('standardized')

        if save_to_pt:
            torch.save(random_n_noises, os.path.join(noises_path, filename))

    return random_n_noises


def create_noisy_traces(dataset: torch.tensor, desired_snr: float, labels: torch.tensor, noise_traces: torch.tensor,
                        calc_snr) -> \
        (list[torch.tensor], list[int]):
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
            noisy_trace = tr + factor * noise
            # noisy_trace = standardize_trace(noisy_trace) if standardize else noisy_trace
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


def get_random_noise_trace(noises_path: str = 'Noises', sampling_rate: float = -1,
                           silent_exception_prints: bool = False) -> torch.tensor:
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


def create_loader_by_phase_and_length(phase_label: str, trace_length: int, targets_path: str, data: WaveformDataset,
                                      batch_size: int) -> (DataLoader, int):
    targets_task23 = pd.read_csv(os.path.join(targets_path, 'task23.csv'))
    merged_metadata = pd.merge(data.metadata, targets_task23, on='trace_name')
    requested_event_list = []
    filtered_metadata = merged_metadata[(merged_metadata.phase_label == phase_label)]
    if requested_event_list:
        filtered_metadata = filtered_metadata[filtered_metadata.source_id.isin(requested_event_list)]
    else:
        print('All events will contribute to the resulting dataset')
    gen = sbg.SteeredGenerator(data, filtered_metadata)
    print(f'Generator contains  {len(gen)} relevant traces')
    augmentations = [
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.SteeredWindow(windowlen=trace_length, strategy="pad")
    ]
    gen.add_augmentations(augmentations)

    @gen.augmentation
    def get_arrival_sample(state_dict):
        _, metadata = state_dict["X"]
        key = f"trace_{state_dict['_control_']['full_phase_label']}_arrival_sample"
        state_dict['station_code'] = (metadata['station_code'], key)
        state_dict["onset_sample"] = (metadata[key], None)

    num_traces = int(len(gen))

    data_loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader, num_traces


def remove_traces_not_to_use(noisy_traces_list: list[torch.tensor], noisy_labels_list: list[torch.tensor],
                             noisy_data_path_list: list[str], num_of_original_traces: int) -> (
        list[torch.tensor], list[torch.tensor], list[int]):
    assert num_of_original_traces > 0, f'Expected positive number of original traces. Got {num_of_original_traces}'
    total_indicies_not_to_use = []
    snr_indices_used = []
    for ndp in noisy_data_path_list:
        snr_indices_not_used = torch.load(os.path.join(ndp, 'indices_not_used')).tolist()
        total_indicies_not_to_use.extend(snr_indices_not_used)
        snr_indices_used.append(
            sublist_complement(containing_list=list(range(num_of_original_traces)), sublist=snr_indices_not_used))
    total_indicies_not_to_use = list(set(total_indicies_not_to_use))

    clean_noisy_traces_list = []
    clean_noisy_labels_list = []
    for i, ndp in enumerate(noisy_data_path_list):
        snr_total_indicies_not_to_use = [k for k, l in enumerate(snr_indices_used[i]) if l in total_indicies_not_to_use]
        snr_total_indicies_to_use = sublist_complement(containing_list=list(range(len(snr_indices_used[i]))),
                                                       sublist=snr_total_indicies_not_to_use)
        clean_noisy_traces_list.append(noisy_traces_list[i][snr_total_indicies_to_use])
        clean_noisy_labels_list.append(noisy_labels_list[i][snr_total_indicies_to_use])

    total_indicies_to_use: list[int] = sublist_complement(containing_list=list(range(num_of_original_traces)),
                                                          sublist=total_indicies_not_to_use)
    return clean_noisy_traces_list, clean_noisy_labels_list, total_indicies_to_use
