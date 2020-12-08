# -*- coding: UTF-8 -*-
"""
Log and other common tool functions
"""
import os
import logging
import numpy as np


logger = None


def init_log_config():
    """
    Initialize config about log
    :return:
    """
    global logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_name = os.path.join(log_path, 'train.log')
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)


def distance_of_pp(p0, p1):
    return np.sqrt(np.sum(np.square(p0 - p1)))


init_log_config()
