#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT4ALL model based on `llama.cpp` backend
"""

import logging
from pathlib import Path
from typing import Callable

__author__ = "abdeladim-s"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"

import logging
import sys
from pyllamacpp.model import Model
from pygptj._logger import set_log_level


class GPT4All(Model):
    """
    GPT4All model

    Example usage
    ```python
    from pygpt4all.models.gpt4all import GPT4All

    def new_text_callback(text):
        print(text, end="")

    model = GPT4All('./models/ggml-gpt4all-j.bin')
    model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
    ```
    """
    _new_text_callback = None

    def __init__(self,
                 model_path: str,
                 log_level: int = logging.INFO):
        """
        :param model_path: The path to a gpt4all `ggml` model
        :param log_level: logging level, set to INFO by default
        """
        # set logging level
        set_log_level(log_level)
        super(GPT4All, self).__init__(ggml_model=model_path, log_level=log_level)

