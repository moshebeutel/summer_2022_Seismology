from snr.calc_snr import CalcSNR, SnrCalcStrategy
from snr.conversions import snr_to_factor
from dataset_creation.noisy_dataset import create_noised_traces_shifted_noise
import torch

from visualization.comparing_plots import plot_experiment_list


def shifting_experiment(trace, full_noise_trace, label: int, model: torch.nn.Module, synthesized_snr: int,
                        num_shifts: int, save_plot_to: str = '', silent_prints: bool = False):
    # Enable print silencing by the caller
    print_fn = lambda string: None if silent_prints else print(string)
    num_samples = trace.shape[-1]
    print_fn(f'Trace contains {num_samples} samples')
    noise_trace = full_noise_trace[:num_samples]

    snr = CalcSNR(SnrCalcStrategy.ENERGY_RATIO)(trace[0].numpy(), label)

    print_fn(f"The original trace of length {trace.shape[1]} has SNR {snr} and onset {label}")
    print_fn(f'full noise trace shape {full_noise_trace.shape}')
    print_fn(f'not shifted noise trace shape {noise_trace.shape}')

    factor = snr_to_factor(trace=trace.clone(), label=label, clean_snr=snr,
                           desired_snr=synthesized_snr, noise_trace=noise_trace.clone(),
                           calc_snr=CalcSNR(SnrCalcStrategy.ENERGY_RATIO))

    print_fn(f'noise_trace multiplied by {factor} added to original trace will result with SNR {synthesized_snr}')
    print_fn('Double Check SNR')
    snr = CalcSNR(SnrCalcStrategy.ENERGY_RATIO)((trace[0] + factor * noise_trace).numpy(), onset=label)
    print_fn(f"The Noised trace of length {trace.shape[1]} has calculated SNR {snr}")

    noised_traces_shifted_noise: list[torch.tensor] = \
        create_noised_traces_shifted_noise(trace=trace, full_noise_trace=full_noise_trace,
                                           factor=factor, num_of_shifts=num_shifts)

    plot_experiment_list(experiment_traces=noised_traces_shifted_noise, label=label, model=model,
                         save_plots_to=save_plot_to)


