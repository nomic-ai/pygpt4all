#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPT4ALL with `llama.cpp` backend through [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp)
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

    model = GPT4All('path/to/gpt4all/model')
    for token in model.generate("Tell me a joke ?"):
        print(token, end='', flush=True)
    ```
    """

    def __init__(self,
                 model_path: str,
                 prompt_context: str = '',
                 prompt_prefix: str = '',
                 prompt_suffix: str = '',
                 log_level: int = logging.ERROR,
                 n_ctx: int = 512,
                 seed: int = 0,
                 n_parts: int = -1,
                 f16_kv: bool = False,
                 logits_all: bool = False,
                 vocab_only: bool = False,
                 use_mlock: bool = False,
                 embedding: bool = False):
        """
        :param model_path: the path to the gpt4all model
        :param prompt_context: the global context of the interaction
        :param prompt_prefix: the prompt prefix
        :param prompt_suffix: the prompt suffix
        :param log_level: logging level, set to INFO by default
        :param n_ctx: LLaMA context
        :param seed: random seed
        :param n_parts: LLaMA n_parts
        :param f16_kv: use fp16 for KV cache
        :param logits_all: the llama_eval() call computes all logits, not just the last one
        :param vocab_only: only load the vocabulary, no weights
        :param use_mlock: force system to keep model in RAM
        :param embedding: embedding mode only
        """
        # set logging level
        set_log_level(log_level)
        super(GPT4All, self).__init__(model_path=model_path,
                                      prompt_context=prompt_context,
                                      prompt_prefix=prompt_prefix,
                                      prompt_suffix=prompt_suffix,
                                      log_level=log_level,
                                      n_ctx=n_ctx,
                                      seed=seed,
                                      n_parts=n_parts,
                                      f16_kv=f16_kv,
                                      logits_all=logits_all,
                                      vocab_only=vocab_only,
                                      use_mlock=use_mlock,
                                      embedding=embedding)

