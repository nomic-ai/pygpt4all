#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT4ALL with `llama.cpp` backend
"""

__author__ = "abdeladim-s"
__copyright__ = "Copyright 2023,"
__license__ = "MIT"

import logging
import pyllamacpp.model
from pygptj._logger import set_log_level


class GPT4All(pyllamacpp.model.Model):
    """
    GPT4All model

    Base: [pyllamacpp.model.Model](#pyllamacpp.model.Model)

    Example usage
    ```python
    from pygpt4all.models.gpt4all import GPT4All

    def new_text_callback(text):
        print(text, end="")

    model = GPT4All('./models/ggml-gpt4all-j.bin')
    model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
    ```
    """

    def __init__(self,
                 model_path: str,
                 n_ctx: int = 512,
                 n_parts: int = -1,
                 seed: int = 0,
                 f16_kv: bool = False,
                 logits_all: bool = False,
                 vocab_only: bool = False,
                 use_mlock: bool = False,
                 embedding: bool = False,
                 log_level: int = logging.INFO):
        """
        :param model_path: The path to a gpt4all `ggml` model
        :param n_ctx: context size
        :param n_parts:
        :param seed: RNG seed, 0 for random
        :param f16_kv: use fp16 for KV cache
        :param logits_all: the llama_eval() call computes all logits, not just the last one
        :param vocab_only: only load the vocabulary, no weights
        :param use_mlock: force system to keep model in RAM
        :param embedding: embedding mode only
        :param log_level: logging level, set to INFO by default
        """
        # set logging level
        set_log_level(log_level)
        super(GPT4All, self).__init__(ggml_model=model_path,
                                      n_ctx=n_ctx,
                                      n_parts=n_parts,
                                      seed=seed,
                                      f16_kv=f16_kv,
                                      logits_all=logits_all,
                                      vocab_only=vocab_only,
                                      use_mlock=use_mlock,
                                      embedding=embedding,
                                      log_level=log_level)

