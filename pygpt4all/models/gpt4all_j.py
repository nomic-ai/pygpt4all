#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT4ALL with `ggml` backend
"""

__author__ = "abdeladim-s"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"

import logging

import pygptj.model
from pygptj._logger import set_log_level


class GPT4All_J(pygptj.model.Model):
    """
    GPT4ALL-J model

    Example usage
    ```python
    from pygpt4all.models.gpt4all_j import GPT4All_J

    model = GPT4All_J('.path/to/gpr4all-j/model')
    for token in model.generate("Tell me a joke ?"):
        print(token, end='', flush=True)
    ```
    """

    def __init__(self,
                 model_path: str,
                 prompt_context: str = '',
                 prompt_prefix: str = '',
                 prompt_suffix: str = '',
                 log_level: int = logging.ERROR):
        """
        :param model_path: The path to a gpt4all-j model
        :param prompt_context: the global context of the interaction
        :param prompt_prefix: the prompt prefix
        :param prompt_suffix: the prompt suffix
        :param log_level: logging level
        """
        # set logging level
        set_log_level(log_level)
        super(GPT4All_J, self).__init__(model_path=model_path,
                                        prompt_context=prompt_context,
                                        prompt_prefix=prompt_prefix,
                                        prompt_suffix=prompt_suffix,
                                        log_level=log_level)
