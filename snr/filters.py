import torch


def autocor(trace):
    from scipy import signal
    autocorr = signal.fftconvolve(trace, trace[::-1], mode='full')
    return autocorr


def filter_fft(input_signal, lower_cut_off, upper_cut_off, sample_rate, band_reject=False):
    # FFT the signal
    sig_fft = torch.fft.fft(input_signal)
    # copy the FFT results
    sig_fft_filtered = sig_fft.clone()

    # obtain the frequencies using scipy function
    freq = torch.fft.fftfreq(input_signal.shape[-1], d=1. / sample_rate)

    filter_mask_fn = lambda f: (torch.abs(f) < lower_cut_off) | (torch.abs(f) > upper_cut_off) \
        if not band_reject else \
        (torch.abs(f) > lower_cut_off) & (torch.abs(f) < upper_cut_off)

    # band-pass filter by assign zeros to the
    # FFT amplitudes where the absolute
    # frequencies smaller than the lower-cut-off or higher than the upper cut-off
    for ch in range(sig_fft_filtered.shape[-2]):
        sig_fft_filtered[ch, filter_mask_fn(freq)] = 0

    # get the filtered signal in time domain
    filtered = torch.fft.ifft(sig_fft_filtered)

    return filtered
