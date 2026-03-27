from typing import Dict, Any, Optional
import random
import logging
import time
import os

import numpy as np
import torch

import sys

# Randomness leaks in from everywhere.
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")
    else:
        print("Using CPU (training will be slow)")
        return torch.device("cpu")

def save_checkpoint(model, optimizer, scheduler, epoch, val_score, fold, cfg, filepath):
    checkpoint =  {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'val_score': val_score,
        'fold': fold,
        'model_name': cfg.model_name,
        'img_size': cfg.img_size,
        'num_classes': cfg.num_classes
    }
    return torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, scheduler = None, optimizer=None, device=torch.device("cpu")):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'val_score': checkpoint['val_score'],
        'fold': checkpoint['fold']
    }


def setup_logging(log_dir, name):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create a logger with name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(name)))
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
