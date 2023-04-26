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
    A simple Python class on top of llama.cpp

    Example usage
    ```python
    def new_text_callback(text):
        print(text, end="")

    model = Model(ggml_model='./models/ggml-model-f16-q4_0.bin', n_ctx=512)
    model.generate("hi my name is ", n_predict=55, new_text_callback=new_text_callback)
    ```
    """
    _new_text_callback = None
    _grab_text_callback = None

    def __init__(self,
                 ggml_model: str,
                 log_level: int = logging.INFO,
                 **llama_params):
        """
        :param ggml_model: the path to the ggml model
        :param log_level: logging level, set to INFO by default
        :param llama_params: keyword arguments for different whisper.cpp parameters,
                        see [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.LLAMA_CONTEXT_PARAMS_SCHEMA)
        """
        # set logging level
        set_log_level(log_level)
        self._ctx = None

        if not Path(ggml_model).is_file():
            raise Exception(f"File {ggml_model} not found!")

        self.llama_params = pp.llama_context_default_params()
        # update llama_params
        self._set_params(self.llama_params, llama_params)

        self._ctx = pp.llama_init_from_file(ggml_model, self.llama_params)

        # gpt params
        self.gpt_params = pp.gpt_params()

        self.res = ""

    @staticmethod
    def _set_params(params, kwargs: dict) -> None:
        """
        Private method to set the kwargs params to the `Params` class
        :param kwargs: dict like object for the different params
        :return: None
        """
        for param in kwargs:
            setattr(params, param, kwargs[param])

    def _call_new_text_callback(self, text) -> bool:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :return: bool (continue generation?)
        """
        # the callback returns either a boolean or a None
        if Model._new_text_callback is not None:
            continue_gen = Model._new_text_callback(text)
            if not(continue_gen is None or continue_gen==True):
                self._ctx.continue_gen = False
        # save res
        self.res += text

    def _call_grab_text_callback(self) -> str:
        if Model._grab_text_callback is not None:
            return Model._grab_text_callback()
        return None

    def num_tokens(self, prompt:str):
        """
        Computes the number of tokens from the prompt text

        :param prompt: the prompt
        :return: the prompt
        """
        return pp.llama_get_nb_tokens(self._ctx, prompt)

    def generate(self, prompt: str,
                 n_predict: int = 128,
                 new_text_callback: Callable[[str], None] = bool,
                 grab_text_callback: Callable[[], str] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """
        Runs llama.cpp inference to generate new text content from the prompt provided as input

        :param prompt: the prompt
        :param n_predict: number of tokens to generate
        :param new_text_callback: a callback function called when new text is generated, default `None`
        :param verbose: print some info about the inference
        :param gpt_params: any other llama.cpp params see [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.GPT_PARAMS_SCHEMA)
        :return: the new generated text
        """
        self.gpt_params.prompt = prompt
        self.gpt_params.n_predict = n_predict
        # update other params if any
        self._set_params(self.gpt_params, gpt_params)

        # assign new_text_callback
        self.res = ""
        Model._new_text_callback = new_text_callback
        Model._grab_text_callback = grab_text_callback

        # run the prediction
        pp.llama_generate(self._ctx, self.gpt_params, self._call_new_text_callback, self._call_grab_text_callback, verbose)
        return self.res

    @staticmethod
    def get_params(params) -> dict:
        """
        Returns a `dict` representation of the params
        :return: params dict
        """
        res = {}
        for param in dir(params):
            if param.startswith('__'):
                continue
            res[param] = getattr(params, param)
        return res

    @staticmethod
    def get_params_schema() -> dict:
        """
        A simple link to [PARAMS_SCHEMA](/pyllamacpp/#pyllamacpp.constants.PARAMS_SCHEMA)
        :return: dict of params schema
        """
        return constants.GPT_PARAMS_SCHEMA

    def __del__(self):
        if self._ctx:
            pp.llama_free(self._ctx)

