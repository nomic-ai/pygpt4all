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

from pygpt4all.models.base_model import BaseModel
from pygpt4all.enums import ModelType


class GPT4All_J(pygptj.model.Model, BaseModel):
    """
    GPT4ALL-J model

    Example usage
    ```python
    from pygpt4all.models.gpt4all_j import GPT4All_J

    def new_text_callback(text):
        print(text, end="")

    model = GPT4All_J('./models/ggml-gpt4all-j.bin')
    model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
    ```
    """

    model_type = ModelType.GPT_J

    def __init__(self,
                 model_path: str,
                 log_level: int = logging.INFO):
        """
        :param model_path: The path to a gpt4all `ggml` model
        :param log_level: logging level, set to INFO by default
        """
        # set logging level
        set_log_level(log_level)
        super(GPT4All_J, self).__init__(model_path=model_path, log_level=log_level)
