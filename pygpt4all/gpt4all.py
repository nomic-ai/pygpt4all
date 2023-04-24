#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python handler for gpt usages standarized for all GPT4All models
"""

__author__ = "rguo123"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

from enum import Enum
import json
import logging
import os

import requests
from tqdm import tqdm

from pygpt4all._logger import set_log_level


# todo: find better name?
class ModelTypes(Enum):
    GPTJ = "gpt_j"
    LLAMA = "llama"


class GPT4All():
    """
    Main handler for all GPT4All models
    """

    def __init__(self,
                 model_path: str = None,
                 log_level: int = logging.INFO):
        """
        :param model_path: The path to a gpt4all `ggml` model
        :param log_level: logging level, set to INFO by default
        """
        # set logging level
        set_log_level(log_level)
        self.model = None


    def _map_gpt4all_model_to_type(self, model_file):
        # Map model filename to correct loader
        pass
    
    @staticmethod
    def list_models():
        response = requests.get('https://gpt4all.io/models/models.json')
        model_json = json.loads(response.content)
        return model_json

    def retrieve_model(self, model_filename=None, download_dir=""):
        def get_download_url(model_filename):
            return f"https://gpt4all.io/models/{model_filename}"

        model_list = self.list_models()
        
        if model_filename is None:
            for item in model_list:
                if "isDefault" in item and item["isDefault"] == "true":
                    model_filename = item["filename"]

        # Before attempting download, check that filename is valid
        model_match = False
        for item in model_list:
            if model_filename == item["filename"]:
                model_match = True
                break

        if not model_match:
            raise ValueError("Invalid model filename")

        download_url = get_download_url(model_filename)
        download_path = os.path.join(download_dir, model_filename)
        
        response = requests.get(download_url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1048576 #1 MB
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(download_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError("ERROR, something went wrong with the download!")
            
        print("Model downloaded at: " + download_path)
        return download_path


    def load_model(self, model=None, **model_params):
        
        # Take model path to load. If not exists, try retrieving model
        # This method will also need to handle llama or gptj
        # Allow for custom models
        pass

    def generate(self, ):
        pass


if __name__ == "__main__":
    gpt4all = GPT4All()
    print(gpt4all.list_models())

    gpt4all.retrieve_model()