import math

import numpy as np

from enum import Enum


class SnrCalcStrategy(Enum):
    ENERGY_RATIO = 1
    MAX_AMPLITUDE_VS_RMS_RATIO = 2
    BANDPASS_FILTER = 3


def str2snr_strategy(str):
    snr_strategy_dict = {'energy_ratio': SnrCalcStrategy.ENERGY_RATIO,
                         'max_amplitude_vs_rms_ratio': SnrCalcStrategy.MAX_AMPLITUDE_VS_RMS_RATIO,
                         'bandpass_filter': SnrCalcStrategy.BANDPASS_FILTER}
    assert str in snr_strategy_dict, f'Wrong strategy name. expected one of \n{snr_strategy_dict.keys()}, \nGot {str}'
    return snr_strategy_dict[str]


class CalcSNR:
    def __init__(self, strategy=SnrCalcStrategy.ENERGY_RATIO):
        self._strategy = strategy
        self._strategy_dict = {
            SnrCalcStrategy.ENERGY_RATIO: CalcSNR._calc_snr_energy,
            SnrCalcStrategy.MAX_AMPLITUDE_VS_RMS_RATIO: CalcSNR._calc_snr_max_amplitude_div_rms,
            SnrCalcStrategy.BANDPASS_FILTER: CalcSNR._calc_snr_bandpass_filter
        }

    def __call__(self, *args, **kwargs):
        assert len(args) + len(kwargs) <= 2, \
            f'Too many arguments to SNR. Expected trace and onset. Got {args}  {kwargs}'
        assert len(args) == 2 or 'onset' in kwargs.keys()
        assert len(args) > 0 or 'trace' in kwargs.keys()
        return self._strategy_dict[self._strategy](*args, **kwargs)

    @staticmethod
    def _calc_snr_bandpass_filter(trace, onset):
        raise NotImplementedError()

    @staticmethod
    def _calc_snr_energy(trace, onset):
        En = np.mean(np.square(trace[:onset]))
        Es = np.mean(np.square(trace[onset:]))
        return CalcSNR.to_db(float(Es / En))

    @staticmethod
    def _calc_snr_max_amplitude_div_rms(trace, onset, window_length=100):
#         half_window = int(window_length / 2.0)
        # print('hwnd',half_window, 'onst', onset)
        max_signal_amp = np.max(np.abs(trace[onset: (onset + window_length)]))
        noise_rms = np.sqrt(np.mean(np.square(trace[:onset])))
        # Convert to db
        # 2 factor because it's a ratio of amplitudes not energies or powers
        return 2 * CalcSNR.to_db(float(max_signal_amp / noise_rms))
        
    @staticmethod
    def to_db(snr):
        return math.log10(snr) * 10

    @staticmethod
    def from_db(snr_db):
        return 10 ** (snr_db / 10)
