import os
import tempfile
from typing import Callable
from ray.tune.result import DEFAULT_RESULTS_DIR
from datetime import datetime
from ray.tune.logger import UnifiedLogger

def get_default_logger_creator(log_dir: str,
                               env_id: str,
                               train_setting: str
                               ) -> Callable:
    if not log_dir.startswith('~/'):
        raise Exception("must start with ~/")
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if train_setting:
        logdir_prefix = "{}_{}_{}_{}".format('PPO', env_id, train_setting, timestr)
    else:
        logdir_prefix = "{}_{}_{}".format('PPO', env_id, timestr)

    def default_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        log_path = os.path.expanduser(log_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=log_path)
        return UnifiedLogger(config, logdir, loggers=None)
    return default_logger_creator