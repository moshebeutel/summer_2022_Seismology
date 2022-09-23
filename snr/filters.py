
def autocor(trace):
    from scipy import signal
    autocorr = signal.fftconvolve(trace, trace[::-1], mode='full')
    return autocorr
