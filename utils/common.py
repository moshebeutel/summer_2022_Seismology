import argparse
import logging
import os
import sys
import time
import random

import numpy as np
import torch


def config_logger(name='default', level=logging.DEBUG, log_folder='../log/'):
    # config logger
    log_format = '%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s'
    formatter = logging.Formatter(log_format)
    logging.basicConfig(level=level,
                        format=log_format,
                        filename=f'{log_folder}{time.ctime()}_{name}.log',
                        filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    created_logger = logging.getLogger(name + '_logger')
    created_logger.addHandler(handler)
    return created_logger


def get_device(cuda=True, gpus='0'):
    return torch.device("cuda:" + gpus if torch.cuda.is_available() and cuda else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :param cudnn_enabled:

    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_batch(batch, model):
    with torch.no_grad():
        pred = model(batch)
        pred = pred.cpu()
    return pred


def max_onset_pred(pred_probs):
    return torch.argmax(pred_probs[:, 0], dim=1)


def shuffle_tensors(tensors: list):
    from more_itertools import random_permutation
    assert len(tensors) > 0, 'List of tensors is empty'
    m = tensors[0].shape[0]
    indices = list(random_permutation(range(m)))
    ret_tensors = [t[indices] for t in tensors if t.shape[0] == m]
    assert len(ret_tensors) == len(tensors)
    return ret_tensors


def load_dataset(filename: str):
    assert filename.endswith('.pt'), f'Expected pt file. Got {filename}'
    if not os.path.exists(filename):
        raise FileExistsError(f'{filename} is not a valid file or file does not exist.')
    return torch.load(filename)


def load_pretrained_model_from_file(filename: str):
    assert filename.endswith('.pth'), f'Expected pth file. Got {filename}'
    if not os.path.exists(filename):
        raise FileExistsError(f'{filename} is not a valid file or file does not exist.')