#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a simple Python API around [llama.cpp](https://github.com/ggerganov/llama.cpp)
"""

import logging
from pathlib import Path
from typing import Callable
import pyllamacpp.constants as constants
from pyllamacpp._logger import set_log_level

__author__ = "abdeladim-s"
__github__ = "https://github.com/abdeladim-s/pyllamacpp"
__copyright__ = "Copyright 2023, "
__license__ = "MIT"

import logging
import sys
import _pyllamacpp as pp


class Model:
    """
    A simple Python class to use llama.cpp

    Example usage
    ```python
    def new_text_callback(text):
        print(text, end="")

    model = Model(ggml_model='./models/ggml-model-f16-q4_0.bin', n_ctx=512)
    model.generate("hi my name is ", n_predict=55, new_text_callback=new_text_callback)
    ```
    """
    _new_text_callback = None

    def __init__(self,
                 ggml_model: str,
                 n_ctx: int = 512,
                 log_level: int = logging.INFO,
                 **params):
        """
        :param ggml_model: the path to the ggml model
        :param n_ctx: The maximum number of tokens of the prompt, default to 512
        :param log_level: logging level, set to INFO by default
        :param params: keyword arguments for different whisper.cpp parameters,
                        see [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.PARAMS_SCHEMA)
        """
        # set logging level
        set_log_level(log_level)

        if not Path(ggml_model).is_file():
            raise Exception(f"File {ggml_model} not found!")
        self._model = pp.llama_model()
        self._vocab = pp.gpt_vocab()
        self.params = pp.gpt_params()
        # init the model
        self._init_model(ggml_model, n_ctx)
        # assign params
        self._set_params(params)

        self.res = ""

    def _init_model(self, model_path: str, n_ctx: int) -> None:
        """
        Private method to set load the model
        :param model_path:  ggml model path
        :param n_ctx: n_ctx
        :return: None
        """
        logging.info("Loading model ...")
        if not pp.llama_model_load(model_path, self._model, self._vocab, n_ctx):
            logging.error(f"failed to load model from {model_path}\n")
            sys.exit(1)

    def _set_params(self, kwargs: dict) -> None:
        """
        Private method to set the kwargs params to the `Params` class
        :param kwargs: dict like object for the different params
        :return: None
        """
        for param in kwargs:
            setattr(self.params, param, kwargs[param])

    def _call_new_text_callback(self, text) -> None:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :return: None
        """
        if Model._new_text_callback is not None:
            Model._new_text_callback(text)
        # save res
        self.res += text

    def generate(self, prompt: str,
                 n_predict: int = 128,
                 new_text_callback: Callable[[str], None] = None,
                 verbose: bool = False,
                 **params) -> str:
        """
        Runs llama.cpp inference to generate new text content from the prompt provided as input

        :param prompt: the prompt
        :param n_predict: number of tokens to generate
        :param new_text_callback: a callback function called when new text is generated, default `None`
        :param verbose: print some info about the inference
        :param params: any other llama.cpp params see [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.PARAMS_SCHEMA)
        :return: the new generated text
        """
        self.params.prompt = prompt
        self.params.n_predict = n_predict
        # update other params if any
        self._set_params(params)

        # assign new_text_callback
        self.res = ""
        Model._new_text_callback = new_text_callback

        # run the prediction
        pp.llama_generate(self.params, self._model, self._vocab, self._call_new_text_callback, verbose)
        return self.res

    def get_params(self) -> dict:
        """
        Returns a `dict` representation of the actual params
        :return: params dict
        """
        res = {}
        for param in dir(self.params):
            if param.startswith('__'):
                continue
            res[param] = getattr(self.params, param)
        return res

    @staticmethod
    def get_params_schema() -> dict:
        """
        A simple link to [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.PARAMS_SCHEMA)
        :return: dict of params schema
        """
        return constants.PARAMS_SCHEMA

    def __del__(self):
        pp.llama_free(self._model)
