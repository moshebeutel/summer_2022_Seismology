import time

import torch
from matplotlib import pyplot as plt

from snr.calc_snr import CalcSNR, SnrCalcStrategy
from utils.common import predict


def plot_compare_fft(sig1, title1, sig2, title2, sample_rate):
    fig, (ax_trace, ax_noise) = plt.subplots(1, 2, figsize=(8, 8), sharey=False)
    N = sig1.shape[-1]
    n = torch.arange(N)
    T = N / sample_rate
    freq = n / T
    ax_trace.set_title(title1)
    ax_trace.stem(freq.numpy(), torch.abs(sig1).numpy(), 'b', markerfmt=' ', basefmt='-b');
    ax_noise.set_title(title2)
    N = sig2.shape[-1]
    n = torch.arange(N)
    T = N / sample_rate
    freq = n / T
    ax_noise.stem(freq.numpy(), torch.abs(sig2).numpy(), 'b', markerfmt=' ', basefmt='-b');


def plot_compare(signals: list[torch.tensor], titles: list[str]):
    assert len(signals) == len(
        titles), f'Expected same amount of signals and titles. Got {len(signals)} signals and {len(titles)} titles'
    fig, axs = plt.subplots(1, int(len(signals)), figsize=(8, 8), sharey='all')
    assert len(axs) == len(signals)
    for ax, s, t in zip(axs, signals, titles):
        ax.set_title(t)
        ax.plot(s)


def plot_experiment_list(experiment_traces: list[torch.tensor], label: int, model: torch.nn.Module, save_plots_to: str):
    num_traces = len(experiment_traces)
    fig, axs = plt.subplots(num_traces, sharey='all', figsize=(8, num_traces * 2))
    fig.suptitle('Merging Noise with Shifts')
    for i, nt in enumerate(experiment_traces):
        axs[i].plot(nt[0])

        snr = CalcSNR(SnrCalcStrategy.ENERGY_RATIO)(nt[0].numpy(), onset=label)

        prediction = predict(nt, model=model)

        axs[i].title.set_text(
            f'Shift {i} seconds. SNR {snr}, Prediction {int(prediction)}'
            f' label {label} residual {int(torch.abs(label - prediction))}')

    fig.tight_layout()
    if save_plots_to:
        folder = os.path.join(save_plots_to, f'shift_experiment_{time.asctime()}/')
        if not os.path.exists(folder):
            os.mkdir(folder)

        plt.savefig(os.path.join(folder, f'shifts.png'))
