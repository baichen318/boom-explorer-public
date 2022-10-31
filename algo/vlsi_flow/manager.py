# Author: baichen318@gmail.com


import os
import logging
from utils import if_exist
from typing import List, NoReturn
from multiprocessing.pool import ThreadPool
from .vlsi_flow import construct_vlsi_manager


class MultiLogHandler(logging.Handler):
    """
        support for multiple loggers
    """
    def __init__(self, dirname):
        super(MultiLogHandler, self).__init__()
        self._loggers = {}
        self._dirname = dirname
        mkdir(self.dirname)

    @property
    def loggers(self):
        return self._loggers

    @property
    def dirname(self):
        return self._dirname

    def flush(self):
        self.acquire()
        try:
            for logger in self.loggers.values():
                logger.flush()
        finally:
            self.release()

    def _get_or_open(self, key):
        self.acquire()
        try:
            if key in self.loggers.keys():
                return self.loggers[key]
            else:
                logger = open(os.path.join(self.dirname, "{}.log".format(key)), 'a')
                self.loggers[key] = logger
                return logger
        finally:
            self.release()

    def emit(self, record):
        try:
            logger = self._get_or_open(record.threadName)
            msg = self.format(record)
            logger.write("{}\n".format(msg))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def create_logger(configs):
    """
        override `create_logger` in utils.py
    """
    import logging
    logger = logging.getLogger()
    head = "[INFO]: %(asctime)-15s %(threadName)12s: %(message)s"
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logging.Formatter(head))
    logging.getLogger().addHandler(stderr_handler)

    # creates a logger per thread
    multi_log_handler = MultiLogHandler(os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                os.path.pardir,
                os.path.pardir,
                configs["vlsi-log"]
            )
        )
    )
    multi_log_handler.setFormatter(logging.Formatter(head))
    logging.getLogger().addHandler(multi_log_handler)
    logger.setLevel(logging.INFO)
    logger.info("create the logger")
    return logger


def vlsi_flow(design_space: object, idx: List[int]) -> NoReturn:
    # parallel VLSI flow
    # create the logger
    logger = create_logger(design_space.configs)
    # execute VLSI flow in parallel
    parallel = design_space.configs["vlsi-flow"]["parallel"]
    p = ThreadPool(parallel)
    i = 0
    for _idx in idx:
        vlsi_manager = construct_vlsi_manager(
            _idx[i],
            design_space.configs["vlsi-flow"]["vlsi-hammer-config"],
            design_space.configs["vlsi-flow"]["benchmarks"]
        )
        p.apply_async(vlsi_manager.run, (logger, _idx[i],))
        i += 1
        if i % parallel == 0:
            p.close()
            p.join()
    p.close()
    p.join()
