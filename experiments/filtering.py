from snr.filters import filter_fft
from evaluation.noisy_dataset_evaluation import predict, get_residual
from visualization.comparing_plots import plot_compare


def fft_filter_experiment(traces_list: list, label_list: list, model, sample_rate: int, residual_threshold_seconds: int,
                          lower_cut_off: int, upper_cut_off: int, plot_fixed_traces=False,
                          silent_prints=True, eval_fn=None):
    assert lower_cut_off < upper_cut_off, f'Expected lower_cut_off < upper_cut_off. ' \
                                          f'Got lower_cut_off = {lower_cut_off} upper_cut_off = {upper_cut_off}'
    # Enable print silencing by the caller
    print_fn = lambda string: None if silent_prints else print(string)
    fixed_counter = 0
    weird_counter = 0
    i = 0
    for noised_trace, le_label in zip(traces_list, label_list):
        filtered = filter_fft(input_signal=noised_trace, lower_cut_off=lower_cut_off, upper_cut_off=upper_cut_off,
                              sample_rate=sample_rate)

        problematic_prediction = predict(noised_trace, model=model, eval_fn=eval_fn)

        filtered_prediction = predict(filtered.float(), model=model, eval_fn=eval_fn)

        problematic_residual = get_residual(problematic_prediction, le_label)

        filtered_residual = get_residual(filtered_prediction, le_label)

        if problematic_residual < residual_threshold_seconds * sample_rate:
            weird_counter += 1
        elif filtered_residual < residual_threshold_seconds * sample_rate:
            fixed_counter += 1
            if plot_fixed_traces:
                plot_compare(signals=[noised_trace[0], filtered[0]],
                             titles=[f'original trace - residual {problematic_residual}',
                                     f'fiiltered trace - residual {filtered_residual}'])

        print_fn(
            f'Noised trace No. {i} problematic prediction = {int(problematic_prediction)},'
            f' filtered prediction = {int(filtered_prediction)}, label = {int(le_label)},'
            f' problematic residual = {problematic_residual} , filtered residual {filtered_residual}')

        i += 1
    print_fn(f'{fixed_counter} traces fixed out of {i} traces. Bty, there were {weird_counter} weird cases')
    return fixed_counter


