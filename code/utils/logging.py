import argparse
import tempfile
import traceback
from functools import partial
from pathlib import Path
import logging

import ignite
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, RunningAverage
from ignite.utils import convert_tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import get_uda2_train_test_loaders, get_model
from utils.tsa import TrainingSignalAnnealing

LOGGING_FORMATTER = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")


def setup_logger(logger, level=logging.INFO):

    if logger.hasHandlers():
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.setLevel(level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(LOGGING_FORMATTER)
    logger.addHandler(ch)


def mlflow_batch_metrics_logging(engine, tag, trainer):
    step = trainer.state.iteration
    for name, value in engine.state.metrics.items():
        mlflow.log_metric("{} {}".format(tag, name), value, step=step)


def mlflow_val_metrics_logging(engine, tag, trainer, metrics):
    step = trainer.state.epoch
    for name in metrics.keys():
        value = engine.state.metrics[name]
        mlflow.log_metric("{} {}".format(tag, name), value, step=step)


def log_tsa(engine, tsa):
    step = engine.state.iteration - 1
    if step % 50 == 0:
        mlflow.log_metric("TSA threshold", tsa.thresholds[step].item(), step=step)
        mlflow.log_metric("TSA selection", engine.state.tsa_log['new_y_pred'].shape[0], step=step)
        mlflow.log_metric("Original X Loss", engine.state.tsa_log['loss'], step=step)
        mlflow.log_metric("TSA X Loss", engine.state.tsa_log['tsa_loss'], step=step)


def log_learning_rate(engine, optimizer):
    step = engine.state.iteration - 1
    if step % 50 == 0:
        lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning rate", lr, step=step)
