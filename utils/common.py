import argparse
import logging
import os
import sys
import time
import random

import numpy as np
import torch
from seisbench import models as sbm
from seisbench.models import SeisBenchModel
from typing import Type

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


def predict(trace, model, eval_fn=None):
    evaluation_fn = eval_batch if eval_fn is None else eval_fn
    return max_onset_pred(evaluation_fn(trace.unsqueeze(dim=0), model=model))


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


def get_residual(prediction: int, label: int)->int:
    return int(torch.abs(prediction-label))


def try_get_saved_pt(filename: str, directory: str)->torch.tensor:
    a = None
    full_file_path = os.path.join(directory, filename)
    print('###' + full_file_path)
    if os.path.exists(full_file_path):
        a = torch.load(full_file_path)
    return a


def load_dataset_and_labels(dataset_path: str, labels_path: str)->(torch.tensor, torch.tensor):
    dataset = torch.load(dataset_path)
    labels = torch.load(labels_path)

    assert dataset.shape[0] == labels.shape[
        0], f'Expected one label for each trace. Got {labels.shape[0]} labels for {dataset.shape[0]} traces.'

    return dataset, labels


def load_pretrained_model(model_class: Type[SeisBenchModel], dataset_trained_on):
    print(f'Working with {model_class} on {str.upper(dataset_trained_on)}')
    model_class_name = str(model_class)
    print(f'Load {model_class_name} pretrained weights')
    pretrained_weights = model_class.list_pretrained(details=False)
    print(f'{model_class_name} pretrained keys', pretrained_weights)
    assert dataset_trained_on in pretrained_weights
    return model_class.from_pretrained(dataset_trained_on)


def standardize_trace(trace: torch.tensor):
    m = trace.float().mean(dim=-1, keepdim=True).unsqueeze(dim=0)
    std = trace.float().std(dim=-1, keepdim=True).unsqueeze(dim=0)
    trace = trace.unsqueeze(dim=0) if trace.dim() == 1 else trace
    standardized = torch.stack([(trace[ch] - m[0, ch]) / std[0, ch] for ch in range(trace.shape[0])], dim=0)
    assert standardized.shape == trace.shape, f'Standardization should not change shape. Got {standardize_trace.shape}'
    return standardized
