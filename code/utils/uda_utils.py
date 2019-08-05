# Variation of UDA

import argparse
import tempfile
import traceback
from functools import partial
from pathlib import Path

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


def prepare_batch(batch, device, non_blocking):
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def compute_supervised_loss(engine,
                            batch,
                            model,
                            cfg,
                            output_transform_model=lambda x: x):

    x, y = prepare_batch(batch, device=cfg['device'], non_blocking=True)
    y_pred = output_transform_model(model(x))

    # Supervised part
    loss = cfg['criterion'](y_pred, y)
    supervised_loss = loss

    if cfg['with_tsa']:
        step = engine.state.iteration - 1
        new_y_pred, new_y = cfg['tsa'](y_pred, y, step=step)
        supervised_loss = cfg['criterion'](new_y_pred, new_y)
        engine.state.tsa_log = {
            "new_y_pred": new_y_pred,
            "loss": loss.item(),
            "tsa_loss": supervised_loss.item()
        }

    return supervised_loss


def compute_unsupervised_loss(engine,
                              batch,
                              model,
                              cfg,
                              output_transform_model=lambda x: x):

    unsup_dp, unsup_aug_dp, back_transf = batch
    unsup_x = convert_tensor(unsup_dp, device=cfg['device'], non_blocking=True)
    unsup_aug_x = convert_tensor(unsup_aug_dp, device=cfg['device'], non_blocking=True)

    # Unsupervised part
    unsup_orig_y_pred = output_transform_model(model(unsup_x)).detach()
    unsup_orig_y_probas = torch.softmax(unsup_orig_y_pred, dim=-1)

    unsup_aug_y_pred = output_transform_model(model(unsup_aug_x))
    unsup_aug_y_probas = torch.log_softmax(unsup_aug_y_pred, dim=-1)
    unsup_aug_y_probas = back_transf(unsup_aug_y_probas)

    consistency_loss = cfg['consistency_criterion'](unsup_aug_y_probas, unsup_orig_y_probas)

    return consistency_loss


def train_update_function(engine,
                          batch,
                          model,
                          optimizer,
                          cfg,
                          train1_unsup_loader_iter,
                          train1_sup_loader_iter,
                          train2_unsup_loader_iter,
                          output_transform_model=lambda x: x):

    model.train()
    optimizer.zero_grad()

    unsup_train_batch = next(train1_unsup_loader_iter)
    train1_unsup_loss = compute_unsupervised_loss(engine,
                                                  unsup_train_batch,
                                                  model,
                                                  cfg,
                                                  output_transform_model=output_transform_model)

    sup_train_batch = next(train1_sup_loader_iter)
    train1_sup_loss = compute_supervised_loss(engine,
                                              sup_train_batch,
                                              model,
                                              cfg)

    unsup_test_batch = next(train2_unsup_loader_iter)
    train2_loss = compute_unsupervised_loss(engine,
                                            unsup_test_batch,
                                            model,
                                            cfg)

    final_loss = train1_sup_loss + cfg['lambda'] * (train1_unsup_loss + train2_loss)
    final_loss.backward()

    optimizer.step()

    return {
        'supervised batch loss': train1_sup_loss,
        'consistency batch loss': train2_loss + train1_unsup_loss,
        'final batch loss': final_loss.item(),
    }


def inference_update_function(engine,
                              batch,
                              model,
                              cfg,
                              output_transform_model=lambda x: x,
                              inference_fn=None):
    model.eval()
    with torch.no_grad():
        x, y = batch
        x = convert_tensor(x, device=cfg['device'], non_blocking=True)
        y = convert_tensor(y, device=cfg['device'], non_blocking=True)

        if inference_fn is not None:
            # Custom inference function
            y_pred = inference_fn(x)
        else:
            # Default inference function (forward the input through the model without further processing)
            y_pred = output_transform_model(model(x))

        return {'x': x[0, :, :, :],
                'y_pred': y_pred,
                'y': y}


def inference_standard(im,
                       model,
                       output_transform_model=lambda x: x,
                       **kwargs):
    """Infer folllowing the TTA method

    Args:
        im: batch image to infer
        model: the model used for inference
        output_transform_model: transform of model output

    Return:
        the predicted mask and the image preprocessed (numpy array)"""

    prediction = output_transform_model(model(im))

    return prediction


def load_params(model,
                optimizer=None,
                model_file='',
                optimizer_file='',
                device_name='cpu'):

    if model_file:
        load_checkpoint = torch.load(model_file, map_location=device_name)
        model.load_state_dict(load_checkpoint)

    if optimizer is not None and optimizer_file:
        load_checkpoint = torch.load(optimizer_file, map_location=device_name)
        optimizer.load_state_dict(load_checkpoint)