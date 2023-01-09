import torch


def snr_to_factor(trace, label, clean_snr, desired_snr, noise_trace, calc_snr, precision=0.0099,
                  insane_factor=100000000):
    assert desired_snr <= int(clean_snr), 'desired_snr should be less than the clean snr'
    factor = 0
    diff = 100
    fix_step = 100000.0
    positive_diff = True
    while abs(diff) > precision and factor < insane_factor:
        # while (factor + fix_step) <= 0 or factor + fix_step >= int(clean_snr):
        #     fix_step /= 2.0
        factor += fix_step
        # print('factor', factor)
        trace_added_noise = torch.clone(trace + (factor * noise_trace))
        calculated_snr = calc_snr(trace_added_noise[0].numpy(), label)
        # print('calculated_snr', calculated_snr)
        diff = calculated_snr - desired_snr
        new_positive_diff = diff > 0

        if new_positive_diff != positive_diff:
            positive_diff = new_positive_diff
            fix_step /= (-2.0)

    return factor

