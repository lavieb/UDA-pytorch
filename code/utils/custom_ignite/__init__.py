from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger as TBLogger


class TensorboardLogger(TBLogger):

    def __init__(self, log_dir):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise RuntimeError("This contrib module requires tensorboardX to be installed. "
                               "Please install it with command: \n pip install tensorboardX")

        self.writer = SummaryWriter(logdir=log_dir)

    def close(self):
        self.writer.close()
