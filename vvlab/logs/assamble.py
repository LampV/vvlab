#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2020-04-07 17:02
@edit time: 2020-04-07 18:10
@FilePath: /vvlab/logs/assamble.py
@desc:
"""


import logging
import logging.config
import logging.handlers

levels = {
    'notset': logging.NOTSET,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

formatters = {
    'none': None,
    "simple": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "deprecation": "%(pathname)s: %(module)s: %(funcName)s: \
            %(lineno)d: DeprecationWarning: %(message)s"
}


def streamHandler():
    return logging.StreamHandler()


def fileHandler(filename='default.log'):
    return logging.FileHandler(filename=filename)


htypes = {
    'stream': streamHandler,
    'file': fileHandler
}


def singleHandler(htype='stream', level='warning',
                  formatter='none', *args, **kwargs):
    # assert params are valid
    checks = {'htype': htypes.keys(), 'level': levels.keys(),
              'formatter': formatters.keys()}
    for param, valids in checks.items():
        valids = list(k for k in valids)
        valids_str = ', '.join(valids[:-1]) + ' or ' + valids[-1]
        if eval(param) not in valids:
            raise ValueError(f'bad {param} (must be {valids_str})')

    # create handler
    creater = htypes[htype]
    if htype == 'file':
        filename = 'default.log' if 'filename' not in kwargs \
                else kwargs['filename']
        handler = creater(filename)
    else:
        handler = creater()

    # set attributes
    handler.setLevel(levels[level])
    if formatter is not None:
        handler.setFormatter(logging.Formatter(formatters[formatter]))

    return handler


def singleLogger(*args, **kwargs):
    logger = logging.getLogger()
    handler = singleHandler(*args, **kwargs)
    logger.addHandler(handler)
    return logger


if __name__ == '__main__':
    logger = singleLogger(htype='stream', formatter='deprecation')
    logger.warning(123)
