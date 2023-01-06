from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, max_error
import torch

class Metrics:
    @staticmethod
    def mae(labels, preds):
        assert labels.shape == preds.shape, f'shapes of labels and predictions do not match ' \
                                            f'{labels.shape} and {preds.shape}'
        return mean_absolute_error(labels, preds)

    @staticmethod
    def mse(labels, preds):
        assert labels.shape == preds.shape, f'shapes of labels and predictions do not match ' \
                                            f'{labels.shape} and {preds.shape}'
        return mean_squared_error(labels, preds)

    @staticmethod
    def med_abs_err(labels, preds):
        assert labels.shape == preds.shape, f'shapes of labels and predictions do not match ' \
                                            f'{labels.shape} and {preds.shape}'
        return median_absolute_error(labels, preds)

    @staticmethod
    def max_error(labels, preds):
        assert labels.shape == preds.shape, f'shapes of labels and predictions do not match ' \
                                            f'{labels.shape} and {preds.shape}'
        return max_error(labels, preds)

    @staticmethod
    def large_error_count(labels, preds, threshold=100):
        assert labels.shape == preds.shape, f'shapes of labels and predictions do not match ' \
                                            f'{labels.shape} and {preds.shape}'
        
        return int(torch.count_nonzero(torch.abs(labels-preds)>threshold))
