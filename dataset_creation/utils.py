import os
import torch
from dataset_creation.noisy_dataset import get_random_noise_window, get_random_noise_trace
from snr.calc_snr import CalcSNR, SnrCalcStrategy
from snr.conversions import snr_to_factor
from utils.common import get_residual, predict, try_get_saved_pt

def standardize_trace(trace: torch.tensor):
    m = trace.float().mean(dim=-1, keepdim=True).unsqueeze(dim=0)
    std = trace.float().std(dim=-1, keepdim=True).unsqueeze(dim=0)
    trace = trace.unsqueeze(dim=0) if trace.dim() == 1 else trace
    standardized = torch.stack([(trace[ch] - m[0, ch]) / std[0, ch] for ch in range(trace.shape[0])], dim=0)
    assert standardized.shape == trace.shape, f'Standardization should not change shape. Got {standardize_trace.shape}'
    return standardized


def find_large_error_traces(dataset: torch.tensor, labels: torch.tensor, model: torch.nn.Module,
                            threshold_samples: int = 100) -> list[int]:
    assert dataset.shape[0] == labels.shape[0]
    large_error_idxs = [i for i in range(dataset.shape[0]) if
                        get_residual(prediction=predict(trace=dataset[i], model=model),
                                     label=labels[i]) > threshold_samples]

    return large_error_idxs


def search_large_errors_given_desired_snr(model: torch.nn.Module, dataset: torch.tensor, labels: torch.tensor, noise_traces: torch.tensor,
                                          desired_snr: int = 10, calc_snr=CalcSNR(SnrCalcStrategy.ENERGY_RATIO)):
    noised_traces_list = create_noisy_traces(calc_snr=calc_snr, dataset=dataset, desired_snr=desired_snr, labels=labels,
                                             noise_traces=noise_traces)

    noisy_dataset = torch.stack(noised_traces_list, dim=0)

    large_error_indexes = find_large_error_traces(dataset=noisy_dataset, labels=labels, model=model)

    return large_error_indexes


def get_n_random_noises(num_noises: int, desired_window_size: int, filename: str, noises_path: str,
                        force_resample: bool = False, save_to_pt: bool = False) -> torch.tensor:
    random_n_noises = None
    if not force_resample:
        random_n_noises = try_get_saved_pt(filename=filename, directory=noises_path)
    if random_n_noises is None:
        # Sample new ones
        random_n_noises = torch.stack([get_random_noise_window(tr=get_random_noise_trace(noises_path=noises_path),
                                                               desired_window_size=desired_window_size) for _ in
                                       range(num_noises)], dim=0)
        if save_to_pt:
            torch.save(random_n_noises, os.path.join(noises_path, filename))

    return random_n_noises


def create_noisy_traces(dataset, desired_snr, labels, noise_traces, calc_snr):
    num_of_traces = dataset.shape[0]
    assert num_of_traces == noise_traces.shape[
        0], f'Expected one noise trace for each trace. Got {noise_traces.shape[0]} noise_traces for' \
            f' {num_of_traces} traces.'
    assert num_of_traces == labels.shape[
        0], f'Expected one label for each trace. Got {labels.shape[0]} labels for {num_of_traces} traces.'
    noised_traces_list = []
    for i in range(num_of_traces):
        # Calculate SNR before merging noise
        tr = dataset[i]
        label = int(labels[i])

        # Take the initial part of noise the size of dataset traces
        noise = noise_traces[i][:dataset.shape[-1]]
        # TODO enable torch tensor in CalcSNR
        # TODO debug or at least deal with infinite factor searches
        clean_snr = calc_snr(tr[0].numpy(), label)
        # Find suitable factor to create the desired SNR
        factor = snr_to_factor(trace=tr.clone(), label=label,
                               noise_trace=noise,
                               clean_snr=clean_snr,
                               desired_snr=desired_snr,
                               calc_snr=calc_snr)
        # Merge the trace with a multiplication of the noise
        # TODO Reduce noise mean
        noisy_trace = tr + factor * noise
        noised_traces_list.append(noisy_trace)
    return noised_traces_list
