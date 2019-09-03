"""This file is a variation of the UDA applied for segmentation task"""

from functools import partial
import logging
import os
import shutil
from enum import Enum

import mlflow
import torch
from torch import nn
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler
from ignite.contrib.handlers.polyaxon_logger import PolyaxonLogger, OutputHandler as plxOutputHandler
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events, Engine, State
from ignite.metrics import RunningAverage, ConfusionMatrix, Loss
from ignite.metrics import IoU, mIoU
from ignite.metrics.confusion_matrix import cmAccuracy, cmPrecision, cmRecall

from polyaxon_client.tracking import get_outputs_path, get_outputs_refs_paths, Experiment
from polyaxon_client.exceptions import PolyaxonClientException

from utils.uda_utils import cycle, train_update_function, load_params, inference_update_function, inference_standard
from utils.logging import mlflow_batch_metrics_logging, mlflow_val_metrics_logging, log_tsa, log_learning_rate, save_prediction, create_save_folders, setup_logger
from utils.tsa import TrainingSignalAnnealing
from utils.metrics import cmFbeta
from utils.custom_ignite import TensorboardLogger


class CustomEvents(Enum):
    """
    Events based on back propagation
    """
    ITERATION_K_COMPLETED = "iteration_k_completed"
    ITERATION_K_STARTED = "iteration_k_started"


def run(train_config, logger, **kwargs):

    logger = logging.getLogger('UDA')
    if getattr(train_config, 'debug', False):
        setup_logger(logger, logging.DEBUG)

    # Set Polyaxon environment if needed
    plx_logger = None
    save_dir = None
    output_experiment_path = None
    try:
        plx_logger = PolyaxonLogger()
        experiment = plx_logger.experiment
        save_dir = get_outputs_path()
        output_experiment_path = get_outputs_refs_paths()
        output_experiment_path = output_experiment_path['experiments'][0] if output_experiment_path else None
        logger.debug("Experiment info: {}".format(experiment.get_experiment_info()))
    except PolyaxonClientException as e:
        logger.warning('Logger Polyaxon : ' + str(e))

    # Path configuration
    saves_dict = getattr(train_config, 'saves', {})

    save_dir = saves_dict.get('save_dir', '') if save_dir is None else save_dir
    log_dir = os.path.join(save_dir, saves_dict.get('log_dir', ''))
    save_model_dir = os.path.join(save_dir, saves_dict.get('model_dir', ''))
    save_prediction_dir = os.path.join(save_dir, saves_dict.get('prediction_dir', ''))
    save_config_dir = os.path.join(save_dir, saves_dict.get('config_dir', ''))
    load_model_file = saves_dict.get('load_model_file', '')
    load_optimizer_file = saves_dict.get('load_optimizer_file', '')

    # Create folders
    create_save_folders(save_dir, saves_dict)

    if output_experiment_path is not None:
        model_dir = saves_dict.get('model_dir', '')
        load_model_file = os.path.join(output_experiment_path, model_dir, load_model_file) if load_model_file else None
        load_optimizer_file = os.path.join(output_experiment_path, model_dir, load_optimizer_file) if load_optimizer_file else None

    num_epochs = getattr(train_config, 'num_epochs')
    num_classes = getattr(train_config, 'num_classes')
    device = getattr(train_config, 'device', 'cpu')

    # Set magical acceleration
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    else:
        assert device == 'cpu', 'CUDA device selected but none is available'

    # Set half precision if required
    use_fp_16 = getattr(train_config, 'use_fp_16', False)

    train1_sup_loader = getattr(train_config, 'train1_sup_loader')
    train1_unsup_loader = getattr(train_config, 'train1_unsup_loader')
    train2_unsup_loader = getattr(train_config, 'train2_unsup_loader')
    test_loader = getattr(train_config, 'test_loader')

    save_interval = saves_dict.get('save_interval', 0)
    n_saved = saves_dict.get('n_saved', 0)

    val_interval = getattr(train_config, 'val_interval', 1)
    pred_interval = getattr(train_config, 'pred_interval', 0)

    model = getattr(train_config, 'model').to(device)

    optimizer = getattr(train_config, 'optimizer')

    criterion = getattr(train_config, 'criterion').to(device)
    consistency_criterion = getattr(train_config, 'consistency_criterion').to(device)

    cm_metric = getattr(train_config,
                        'cm_metric',
                        ConfusionMatrix(num_classes=num_classes, output_transform=lambda x: (x['y_pred'], x['y'])))

    # AMP initialization for half precision
    if use_fp_16:
        assert 'cuda' in device
        assert torch.backends.cudnn.enabled, "NVIDIA/Apex:Amp requires cudnn backend to be enabled."
        try:
            from apex import amp
        except:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
        # Initialize amp
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level="O2")

    # Load checkpoint
    load_params(model, optimizer=optimizer, model_file=load_model_file, optimizer_file=load_optimizer_file, device_name=device)

    # Add batch norm
    is_bn = getattr(train_config, 'is_bn', False)
    if is_bn:
        batch_norm = nn.BatchNorm2d(3).to(device)
        if use_fp_16:
            batch_norm = amp.initialize(batch_norm)
        batch_norm.reset_parameters()
        model = nn.Sequential(batch_norm, model)

    # Copy the config file
    shutil.copy2(os.path.abspath(train_config.__file__), os.path.join(save_config_dir, 'checkpoint_module.py'))

    le = len(train1_sup_loader)
    num_train_steps = le * num_epochs
    mlflow.log_param("num train steps", num_train_steps)

    lr = getattr(train_config, 'learning_rate')
    num_warmup_steps = getattr(train_config, 'num_warmup_steps', 0)

    lr_scheduler = getattr(train_config, 'lr_scheduler', None)

    if num_warmup_steps > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler,
                                                       warmup_start_value=0.0,
                                                       warmup_end_value=lr * (1.0 + 1.0 / num_warmup_steps),
                                                       warmup_duration=num_warmup_steps)

    train1_sup_loader_iter = cycle(train1_sup_loader)
    train1_unsup_loader_iter = cycle(train1_unsup_loader)
    train2_unsup_loader_iter = cycle(train2_unsup_loader)

    # Reduce on plateau
    reduce_on_plateau = getattr(train_config, 'reduce_on_plateau', None)

    # Output transform model
    output_transform_model = getattr(train_config, 'output_transform_model', lambda x: x)

    inference_fn = getattr(train_config, 'inference_fn', inference_standard)

    lam = getattr(train_config, 'consistency_lambda')

    tsa = TrainingSignalAnnealing(num_steps=num_train_steps,
                                  min_threshold=getattr(train_config, 'TSA_proba_min'),
                                  max_threshold=getattr(train_config, 'TSA_proba_max'))

    with_tsa = getattr(train_config, 'with_TSA', False)

    cfg = {'tsa': tsa,
           'lambda': lam,
           'with_tsa': with_tsa,
           'device': device,
           'consistency_criterion': consistency_criterion,
           'criterion': criterion}

    trainer = Engine(partial(train_update_function,
                             model=model,
                             optimizer=optimizer,
                             cfg=cfg,
                             train1_sup_loader_iter=train1_sup_loader_iter,
                             train1_unsup_loader_iter=train1_unsup_loader_iter,
                             train2_unsup_loader_iter=train2_unsup_loader_iter,
                             output_transform_model=output_transform_model,
                             use_fp_16=use_fp_16))

    # Register events
    for e in CustomEvents:
        State.event_to_attr[e] = 'iteration'

    trainer.register_events(*CustomEvents)

    if with_tsa:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, log_tsa, tsa)

    if lr_scheduler is not None:
        if not hasattr(lr_scheduler, "step"):
            trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
        else:
            trainer.add_event_handler(Events.ITERATION_STARTED, lambda engine: lr_scheduler.step())

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_learning_rate, optimizer)

    metric_names = [
        'supervised batch loss',
        'consistency batch loss',
        'final batch loss'
    ]

    def output_transform(x, name):
        return x[name]

    for n in metric_names:
        RunningAverage(output_transform=partial(output_transform, name=n)).attach(trainer, n)

    ProgressBar(persist=True, bar_format="").attach(trainer,
                                                    event_name=Events.EPOCH_STARTED,
                                                    closing_event_name=Events.COMPLETED)

    # Handlers for Tensorboard logging
    tb_logger = TensorboardLogger(log_dir=log_dir)
    tb_logger.attach(trainer,
                     log_handler=tbOutputHandler(tag="train",
                                                 metric_names=metric_names),
                     event_name=CustomEvents.ITERATION_K_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=tbOptimizerParamsHandler(optimizer, param_name="lr"),
                     event_name=CustomEvents.ITERATION_K_STARTED)

    # Handlers for Polyaxon logging
    if plx_logger is not None:
        plx_logger.attach(trainer,
                          log_handler=plxOutputHandler(tag="train",
                                                       metric_names=metric_names),
                          event_name=CustomEvents.ITERATION_K_COMPLETED)

    metrics = {'loss': Loss(criterion, output_transform=lambda x: (x['y_pred'], x['y'])),
               'mAcc': cmAccuracy(cm_metric).mean(),
               'mPr': cmPrecision(cm_metric).mean(),
               'mRe': cmRecall(cm_metric).mean(),
               'mIoU': mIoU(cm_metric),
               'mF1': cmFbeta(cm_metric, 1).mean()}
    iou = IoU(cm_metric)
    for i in range(num_classes):
        key_name = 'IoU_{}'.format(str(i))
        metrics[key_name] = iou[i]

    inference_update_fn = partial(inference_update_function,
                                  model=model,
                                  cfg=cfg,
                                  output_transform_model=output_transform_model,
                                  inference_fn=inference_fn)

    evaluator = Engine(inference_update_fn)
    train_evaluator = Engine(inference_update_fn)

    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)
        metric.attach(evaluator, name)

    # Add checkpoint
    if save_model_dir:
        checkpoint = ModelCheckpoint(dirname=save_model_dir,
                                     filename_prefix='checkpoint',
                                     save_interval=save_interval,
                                     n_saved=n_saved,
                                     create_dir=True)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {'mymodel': model,
                                                                       'optimizer': optimizer})

    def trigger_k_iteration_started(engine, k):
        if engine.state.iteration % k == 0:
            engine.fire_event(CustomEvents.ITERATION_K_STARTED)

    def trigger_k_iteration_completed(engine, k):
        if engine.state.iteration % k == 0:
            engine.fire_event(CustomEvents.ITERATION_K_COMPLETED)

    def run_validation(engine, validation_interval):
        if (trainer.state.epoch - 1) % validation_interval == 0:
            train_evaluator.run(train1_sup_loader)
            evaluator.run(test_loader)

            if save_prediction_dir:
                train_output = train_evaluator.state.output
                test_output = evaluator.state.output

                iteration = str(trainer.state.iteration)
                epoch = str(trainer.state.epoch)

                save_prediction('train_{}_{}'.format(iteration, epoch),
                                save_prediction_dir,
                                train_output['x'],
                                torch.argmax(train_output['y_pred'][0, :, :, :], dim=0),
                                y=train_output['y'][0, :, :])

                save_prediction('test_{}_{}'.format(iteration, epoch),
                                save_prediction_dir,
                                test_output['x'],
                                torch.argmax(test_output['y_pred'][0, :, :, :], dim=0),
                                y=test_output['y'][0, :, :])

            train_evaluator.state.output = None
            evaluator.state.output = None

            if reduce_on_plateau is not None:
                reduce_on_plateau.step(evaluator.state.metrics['mIoU'])

    trainer.add_event_handler(Events.ITERATION_STARTED, trigger_k_iteration_started, k=10)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, trigger_k_iteration_completed, k=10)

    trainer.add_event_handler(Events.EPOCH_STARTED, run_validation, validation_interval=val_interval)
    trainer.add_event_handler(Events.COMPLETED, run_validation, validation_interval=1)

    def trainer_prediction_save(engine, prediction_interval):
        if (engine.state.iteration - 1) % prediction_interval == 0:

            if save_prediction_dir:
                trainer_output = trainer.state.output['unsup pred']

                iteration = str(trainer.state.iteration)
                epoch = str(trainer.state.epoch)

                save_prediction('trainer_{}_{}'.format(iteration, epoch),
                                save_prediction_dir,
                                trainer_output['x'],
                                trainer_output['y_pred'])

                logger.debug('Saved trainer prediction for iteration {}'.format(str(engine.state.iteration)))

            trainer.state.output = None

    trainer.add_event_handler(Events.ITERATION_COMPLETED, trainer_prediction_save, prediction_interval=pred_interval)

    tb_logger.attach(train_evaluator,
                     log_handler=tbOutputHandler(tag="train",
                                                 metric_names=list(metrics.keys())),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=tbOutputHandler(tag="test",
                                                 metric_names=list(metrics.keys())),
                     event_name=Events.EPOCH_COMPLETED)

    # Handlers for Polyaxon logging
    if plx_logger is not None:
        plx_logger.attach(train_evaluator,
                          log_handler=plxOutputHandler(tag="train",
                                                       metric_names=list(metrics.keys())),
                          event_name=Events.EPOCH_COMPLETED)

        plx_logger.attach(evaluator,
                          log_handler=plxOutputHandler(tag="test",
                                                       metric_names=list(metrics.keys())),
                          event_name=Events.EPOCH_COMPLETED)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, mlflow_batch_metrics_logging, "train", trainer)
    train_evaluator.add_event_handler(Events.COMPLETED, mlflow_val_metrics_logging, "train", trainer)
    evaluator.add_event_handler(Events.COMPLETED, mlflow_val_metrics_logging, "test", trainer)

    data_steps = list(range(len(train1_sup_loader)))

    logger.debug('Start training')
    trainer.run(data_steps, max_epochs=num_epochs)
    logger.debug('Finished training')
