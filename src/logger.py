import tensorboardX as tbx

class Logger(object):
    """Tensorboard logger.
        tensorboard --logdir=. --port=5000
    """

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tbx.SummaryWriter(log_dir)