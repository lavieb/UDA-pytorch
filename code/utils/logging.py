from PIL import Image
import logging
import os
import numpy as np
from image_dataset_viz import render_datapoint

import mlflow
import torch


LOGGING_FORMATTER = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")


def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _getvocpallete(256)


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
        try:
            mlflow.log_metric("{} {}".format(tag, name), value, step=step)
        except BaseException as e:
            pass


def mlflow_val_metrics_logging(engine, tag, trainer):
    step = trainer.state.epoch
    for name, value in engine.state.metrics.items():
        try:
            mlflow.log_metric("{} {}".format(tag, name), value, step=step)
        except BaseException as e:
            pass


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


def save_prediction(tag, prediction_dir, x, y_pred, y=None):
    """Save prediction in the prediction folder"""
    x = x.cpu().detach().numpy().transpose((1, 2, 0)) if torch.is_tensor(x) else x
    y_pred = y_pred.cpu().detach().numpy() if torch.is_tensor(y_pred) else y_pred

    output_file = os.path.join(prediction_dir,
                               "{}_prediction.jpg".format(tag))

    if y is not None:
        y = y.cpu().detach().numpy() if torch.is_tensor(y) else y

        write_prediction_on_image(mask=y,
                                  mask_predicted=y_pred,
                                  im=x,
                                  filepath=output_file)
    else:
        write_prediction_on_image2(mask_predicted=y_pred,
                                   im=x,
                                   filepath=output_file)


def render_x(im):
    q_min = 0
    q_max = 100

    vmin = np.percentile(im, q=q_min, axis=(0, 1), keepdims=True)
    vmax = np.percentile(im, q=q_max, axis=(0, 1), keepdims=True)
    x = np.clip(im, a_min=vmin, a_max=vmax)
    x = (x - vmin) / (vmax - vmin + 1e-10) * 255
    x = x.astype(np.uint8)

    return x


def write_prediction_on_image(mask, mask_predicted, im, filepath):
    assert isinstance(mask, np.ndarray) and mask.ndim == 2, \
        "{} and {}".format(type(mask), mask.shape if isinstance(mask, np.ndarray) else None)

    assert isinstance(mask_predicted, np.ndarray) and mask_predicted.ndim == 2, \
        "{} and {}".format(type(mask_predicted), mask_predicted.shape if isinstance(mask_predicted, np.ndarray) else None)

    assert isinstance(im, np.ndarray) and im.ndim == 3, \
        "{} and {}".format(type(im), im.shape if isinstance(im, np.ndarray) else None)

    # Normalize for rendering
    x = render_x(im)

    # Save the images and masks
    im = Image.fromarray(x).convert('RGB')

    pil_gt = Image.fromarray(mask.astype('uint8'))
    pil_gt.putpalette(vocpallete)
    pil_gt = pil_gt.convert('RGB')
    res_gt = render_datapoint(im, pil_gt)

    pil_pred = Image.fromarray(mask_predicted.astype('uint8'))
    pil_pred.putpalette(vocpallete)
    pil_pred = pil_pred.convert('RGB')
    res_pred = render_datapoint(im, pil_pred)

    size_image = (mask.shape[0], mask.shape[1])

    tiles = [[im, res_gt, pil_gt], [im, res_pred, pil_pred]]

    nb_cols = len(tiles)
    nb_rows = len(tiles[0])

    cvs = Image.new('RGB', (nb_cols * size_image[0], nb_rows * size_image[1]))
    for i_row in range(nb_cols):
        for i_col in range(nb_rows):
            px, py = (i_row * size_image[0], i_col * size_image[1])
            cvs.paste(tiles[i_row][i_col], (px, py))

    cvs.save(filepath)


def write_prediction_on_image2(mask_predicted, im, filepath):
    assert isinstance(mask_predicted, np.ndarray) and mask_predicted.ndim == 2, \
        "{} and {}".format(type(mask_predicted), mask_predicted.shape if isinstance(mask_predicted, np.ndarray) else None)

    assert isinstance(im, np.ndarray) and im.ndim == 3, \
        "{} and {}".format(type(im), im.shape if isinstance(im, np.ndarray) else None)

    # Normalize for rendering
    x = render_x(im)

    # Save the images and masks
    im = Image.fromarray(x).convert('RGB')

    pil_pred = Image.fromarray(mask_predicted.astype('uint8'))
    pil_pred.putpalette(vocpallete)
    pil_pred = pil_pred.convert('RGB')
    res_pred = render_datapoint(im, pil_pred)

    size_image = (mask_predicted.shape[0], mask_predicted.shape[1])

    tiles = [[im, res_pred, pil_pred]]

    nb_cols = len(tiles)
    nb_rows = len(tiles[0])

    cvs = Image.new('RGB', (nb_cols * size_image[0], nb_rows * size_image[1]))
    for i_row in range(nb_cols):
        for i_col in range(nb_rows):
            px, py = (i_row * size_image[0], i_col * size_image[1])
            cvs.paste(tiles[i_row][i_col], (px, py))

    cvs.save(filepath)


def create_save_folders(save_dir, saves_dict):

    log_dir = os.path.join(save_dir, saves_dict.get('log_dir', ''))
    save_model_dir = os.path.join(save_dir, saves_dict.get('model_dir', ''))
    save_prediction_dir = os.path.join(save_dir, saves_dict.get('prediction_dir', ''))
    save_config_dir = os.path.join(save_dir, saves_dict.get('config_dir', ''))

    if save_dir and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if log_dir and not os.path.exists(log_dir):
        os.mkdir(log_dir)

    if save_model_dir and not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    if save_prediction_dir and not os.path.exists(save_prediction_dir):
        os.mkdir(save_prediction_dir)

    if save_config_dir and not os.path.exists(save_config_dir):
        os.mkdir(save_config_dir)
