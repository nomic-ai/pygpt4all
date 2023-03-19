#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Constants
"""
import logging

PACKAGE_NAME = 'pyllamacpp'

LOGGING_LEVEL = logging.INFO


PARAMS_SCHEMA = {
    'seed': {
            'type': int,
            'description': "RNG seed",
            'options': None,
            'default': -1
    },
    'n_predict': {
            'type': int,
            'description': "Number of tokens to predict",
            'options': None,
            'default': 50
    },
    'n_threads': {
            'type': int,
            'description': "Number of threads",
            'options': None,
            'default': 4
    },
    'repeat_last_n': {
            'type': int,
            'description': "Last n tokens to penalize",
            'options': None,
            'default': 64
    },
    # sampling params
    'top_k': {
            'type': int,
            'description': "top_k",
            'options': None,
            'default': 40
    },
    'top_p': {
            'type': float,
            'description': "top_p",
            'options': None,
            'default': 0.95
    },
    'temp': {
            'type': float,
            'description': "temp",
            'options': None,
            'default': 0.8
    },
    'repeat_penalty': {
            'type': float,
            'description': "repeat_penalty",
            'options': None,
            'default': 1.3
    },
    'n_batch': {
            'type': int,
            'description': "batch size for prompt processing",
            'options': None,
            'default': True
    }
}
