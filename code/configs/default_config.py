import argparse
import tempfile
import traceback
from functools import partial
from pathlib import Path
import logging
import os

import ignite
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler
from ignite.contrib.handlers.polyaxon_logger import PolyaxonLogger, OutputHandler as plxOutputHandler
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, RunningAverage
from ignite.utils import convert_tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_uda_train_test_loaders, get_model
from utils.tsa import TrainingSignalAnnealing

from polyaxon_client.tracking import Experiment, get_outputs_path, get_outputs_refs_paths
from polyaxon_client.exceptions import PolyaxonClientException

from utils.uda_utils import cycle, train_update_function, compute_supervised_loss, compute_unsupervised_loss
from utils.logging import mlflow_batch_metrics_logging, mlflow_val_metrics_logging, log_tsa, log_learning_rate, setup_logger

device = "cuda"

config = {
    "dataset": dataset_name,
    "data_path": ".",

    "model": network_name,

    "momentum": 0.9,
    "weight_decay": 1e-4,
    "batch_size": batch_size,
    "unlabelled_batch_size": 320,
    "num_workers": 10,

    "num_epochs": num_epochs,

    "learning_rate": 0.03,
    "min_lr_ratio": 0.004,
    "num_warmup_steps": 0,

    "num_labelled_samples": 4000,
    "consistency_lambda": 1.0,
    "consistency_criterion": "KL",

    "with_TSA": False,
    "TSA_proba_min": 0.1,
    "TSA_proba_max": 1.0,
}

batch_size = 64
num_epochs = 200

# Save configuration

