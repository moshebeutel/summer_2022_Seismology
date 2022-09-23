# seisbench
# import seisbench
import argparse
import logging

# plotting
import matplotlib.pyplot as plt
# torch and numpy
import numpy as np
import torch

from snr.calc_snr import CalcSNR
from snr.conversions import snr_to_factor
from utils.common import config_logger, set_seed, eval_batch, max_onset_pred


# import wandb as wandb


def main(args):
    batch = torch.load(args.base_dir + 'batch.pt')
    SNR = np.load(args.base_dir + 'SNR.npy')
    pred_probs = torch.load(args.base_dir + 'pred_probs.pt')
    noise_trace = torch.load(args.base_dir + 'noise_trace.pt')
    phasenet_model = torch.load(args.base_dir + 'phasenet_model.pth')
    noise_energy = torch.mean(torch.square(noise_trace))

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
    logger.info(f'Noise of length {noise_trace.shape[1]} with energy {noise_energy} is going to be added')

    trace_idx = args.trace_index
    assert 0 <= trace_idx < high_SNR_traces_indices.shape[0], 'There is no such index in the high SNRs traces'
    trace_batch_idx = high_SNR_traces_indices[trace_idx]
    logger.info(f'Examine trace No. {trace_batch_idx} in batch')

    trc = torch.clone(high_SNR_traces[trace_idx])
    original_trace_snr = SNR_of_high_SNR_traces[trace_idx]
    label = int(high_SNR_traces_labels[trace_idx].int())
    residuals = max_onset_pred(pred_probs) - label
    num_of_levels = int(original_trace_snr) + 1

    # init fig
    fig = plt.figure(figsize=(20, 4 * num_of_levels))
    axs = fig.subplots(num_of_levels + 1, 3, sharex=True, gridspec_kw={"hspace": 0.5})

    # plot original signal
    plot_num = 0
    axs[plot_num, 0].plot(trc.T)
    axs[plot_num, 0].title.set_text(f'Original Signal actual SNR {original_trace_snr}')
    axs[plot_num, 1].plot(noise_trace.T)
    axs[plot_num, 1].title.set_text('Original Noise Trace')
    axs[plot_num, 2].plot(pred_probs[trace_idx, :, :].T)
    axs[plot_num, 2].title.set_text(f'Original Prediction Graph, Residual {residuals[trace_idx]}')
    plot_num += 1

    noisy_traces = []
    noises_added = []
    calc_snr = CalcSNR(strategy=args.snr_calc_strategy)
    for des_snr in reversed(range(num_of_levels)):
        factor = snr_to_factor(trace=trc, label=label, clean_snr=original_trace_snr,
                               desired_snr=des_snr, noise_trace=noise_trace,
                               calc_snr=calc_snr) \
            if des_snr < original_trace_snr \
            else 0

        orig_trc = torch.clone(trc)
        noise_added = torch.clone(factor * noise_trace)
        trace_added_noise = orig_trc + noise_added

        noisy_traces.append(trace_added_noise)
        noises_added.append(noise_added)

        # plot the noised signal
        axs[plot_num, 0].plot(trace_added_noise.T)
        axs[plot_num, 0].title.set_text(
            f'SNR {des_snr} factor {np.round(factor, 3)}' if factor > 0
            else f'Original Signal actual SNR {original_trace_snr}')

        plot_num += 1

    # prepare traces for evaluation
    noisy_trcs_stacked = torch.stack(noisy_traces)
    noisy_preds = eval_batch(noisy_trcs_stacked, phasenet_model)
    # compute residuals based on model predictions
    residuals = max_onset_pred(noisy_preds) - label

    # plot the added noises and the prediction distribution
    for i in range(1, num_of_levels + 1):
        axs[i, 1].plot(noises_added[i - 1].T)
        axs[i, 1].title.set_text('Noise Added')
        axs[i, 2].plot(noisy_preds[i - 1, :, :].T)
        axs[i, 2].title.set_text(f'Prediction Graph, Residual {residuals[i - 1]}')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyse Noised Signal")
    # parser.add_argument("--raw", type=str2bool, default=True, help="Work on raw data or preprocessed features")
    parser.add_argument("--base_dir", type=str,
                        default='/home/moshe/Documents/Helmholtz Summer 2022 Internship/Academic/code/ETHZ_phasenet/')
    parser.add_argument("--dataset", type=str, default='ethz',
                        choices=['ethz', 'geofon', 'stead', 'instance', 'iquique', 'lendb', 'neic', 'scdecs'])
    parser.add_argument("--model", type=str, default='phasenet',
                        choices=['PhaseNet', 'EQTransformer', 'GPD', 'CRED', 'BasicPhaseAE'])
    # parser.add_argument("--from_pretrained", type=str, default='ethz',
    #                     choices=['original', 'ethz', 'geofon', 'stead', 'instance', 'iquique', 'lendb', 'neic',
    #                              'scdecs'])
    parser.add_argument("--seed", type=int, default=42, help="predefined seed value for reproducibility")

    parser.add_argument("--noise_source", type=str, default='dataset_start',
                        choices=['dataset_start', 'trace_start'])
    parser.add_argument("--snr_calc_strategy", type=str, default='energy_ratio',
                        choices=['energy_ratio', 'max_amplitude_vs_rms_ratio', 'bandpass_filter'])
    parser.add_argument("--snr_threshold", type=int, default=10, help="dB value. Seek for SNRs above this threshold")
    parser.add_argument("--trace_index", type=int, default=2, help="The index of the trace in the high snr traces list")
    # pn_model = sbm.PhaseNet.from_pretrained("geofon")

    args = parser.parse_args()

    log_level = logging.INFO

    config_logger(level=log_level)
    logging.getLogger().setLevel(log_level)
    logging.info(f'Logger set. Log level  = {logging.getLevelName(logging.getLogger().getEffectiveLevel())}')
    set_seed(args.seed)

    exp_name = f'Analyze Different Noise Level Predictions -' \
               f' Dataset_{args.dataset} Noise Added From {args.noise_source} ' \
               f'SNR Calculation Function {args.snr_calc_strategy}'
    # Picker Model {args.model} Pretrained  {args.f d}\
    logging.info(f'Experiment Name : {exp_name}')

    # Weights & Biases
    # if args.wandb:
    #     wandb.init(project="seisbench_uncertainty", entity="emg_diff_priv", name=exp_name)
    #     wandb.config.update(args)

    main(args)
