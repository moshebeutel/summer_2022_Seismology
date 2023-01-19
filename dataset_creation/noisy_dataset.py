import torch
from dataset_creation.utils import create_noisy_traces
from snr.calc_snr import CalcSNR, SnrCalcStrategy
from utils.common import sublist_complement


def create_single_noisy_version(original_traces: torch.tensor, original_labels: torch.tensor,
                                augmented_noise_traces: torch.tensor, desired_snr: float,
                                snr_strategy: SnrCalcStrategy = SnrCalcStrategy.ENERGY_RATIO):

    num_original_traces = original_traces.shape[0]
    num_samples_original_trace = original_traces.shape[-1]

    assert original_labels.shape[0] == num_original_traces, f'Expected {num_original_traces} labels'
    assert augmented_noise_traces.shape[0] == num_original_traces, f'Expected {num_original_traces} noise traces'
    assert augmented_noise_traces.shape[-1] >= num_samples_original_trace, \
        f'Expected at least {num_samples_original_trace} ' \
        f'samples in noise traces. Got {augmented_noise_traces.shape[-1]}'

    # Trim the noise traces. Adjust to dataset num samples.
    noise_traces: torch.tensor = augmented_noise_traces[:, :, :num_samples_original_trace].clone()
    print(f'Trimmed {noise_traces.shape[0]} noise traces to shape {noise_traces.shape}')

    # Create the noisy traces and get the ones that did not succeed
    version_noised_traces_list, version_noise_factors, version_not_included_indices = \
        create_noisy_traces(dataset=original_traces, desired_snr=desired_snr, labels=original_labels,
                            noise_traces=noise_traces, calc_snr=CalcSNR(snr_strategy))
    print(f'Created {len(version_noised_traces_list)} noisy traces')
    print(f'The following indices are not included {version_not_included_indices}')

    # Remove the corresponding indices from the full noises list and the label list
    included_indices_list = sublist_complement(containing_list=list(range(num_original_traces)),
                                               sublist=version_not_included_indices)
    version_labels = original_labels[included_indices_list].clone()
    version_augmented_noise_traces = augmented_noise_traces[included_indices_list].clone()

    return torch.stack(version_noised_traces_list, dim=0), version_labels, version_augmented_noise_traces, \
        torch.tensor(version_noise_factors), version_not_included_indices
