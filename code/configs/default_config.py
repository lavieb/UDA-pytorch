import argparse
import tempfile
import traceback
from functools import partial
from pathlib import Path
import logging
import os

import albumentations as albu
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
from utils.autoaugment import SubBackwardPolicy, ImageNetBackwardPolicy
from utils.transforms import ToTensor

# Transformations


def train_transform_fn(im, mask):
    autotransf = ImageNetBackwardPolicy()

    albu_train_transform = albu.Compose([autotransf,
                                         ToTensor()])
    albu_train_transform_fn = lambda dp: albu_train_transform(**dp)

    dp = albu_train_transform_fn({'image': im,
                                  'mask': mask})

    return dp['image'], dp['mask']


def test_transform_fn(im, mask):
    albu_test_transform = albu.Compose([ToTensor()])
    albu_test_transform_fn = lambda dp: albu_test_transform(**dp)

    dp = albu_test_transform_fn({'image': im,
                                 'mask': mask})

    return dp['image'], dp['mask']


def unsup_transform_fn(im, mask):
    autotransf = ImageNetBackwardPolicy()

    albu_unsup_transform = albu.Compose([autotransf,
                                         ToTensor()])
    albu_unsup_transform_fn = lambda dp: albu_unsup_transform(**dp)

    dp = albu_unsup_transform_fn({'image': im,
                                  'mask': mask})

    return dp['image'], dp['mask'], autotransf


debug = True
device = "cuda"

num_epochs = 200
num_classes = 2
val_interval = 1

# Save configuration
saves = {'save_dir': '',
         'log_dir': '',
         'model_dir': '',
         'prediction_dir': '',
         'config_dir': '',
         'save_interval': 0,
         'n_saved': 0
         }

# Datasets
train_set = None
test_set = None

# Dataloader config
batch_size = 64
num_labelled_samples = 4000
unlabelled_batch_size = 320

train_workers = 10
test_workers = 10

pin_memory = True

train1_sup_loader, train1_unsup_loader, train2_unsup_loader, test_loader = \
    get_uda_train_test_loaders(train_set=train_set,
                               test_set=test_set,
                               train_transform_fn=train_transform_fn,
                               test_transform_fn=test_transform_fn,
                               unsup_transform_fn=unsup_transform_fn,
                               num_classes=num_classes,
                               num_labelled_samples=num_labelled_samples,
                               batch_size=batch_size,
                               train_workers=train_workers,
                               test_workers=test_workers,
                               unlabelled_batch_size=unlabelled_batch_size,
                               pin_memory=pin_memory)

# Model
model = None

# Criterions
criterion = nn.CrossEntropyLoss()
consistency_criterion = nn.KLDivLoss(reduction='batchmean')

consistency_lambda = 1.0

# Optimizer
learning_rate = 0.03
optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate,
                      momentum=0.9,
                      weight_decay=1e-4,
                      nesterov=True)

# LR scheduler
min_lr_ratio = 0.004
num_warmup_steps = 0
eta_min = learning_rate * min_lr_ratio

num_train_steps = len(train1_sup_loader) * num_epochs
lr_scheduler = CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=num_train_steps - num_warmup_steps)

# TSA
with_TSA = False
TSA_proba_min = 0.1
TSA_proba_max = 1.0
