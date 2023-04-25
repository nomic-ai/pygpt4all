#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python handler for gpt usages standarized for all GPT4All models
"""

__author__ = "rguo123"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

import json
import logging
import os

import requests
from tqdm import tqdm

from pygpt4all._logger import set_log_level
from pygpt4all.enums import ModelType


class GPT4All:
    """
    Main handler for all GPT4All models
    """

    def __init__(self, log_level: int = logging.INFO):
        """
        :param log_level: logging level, set to INFO by default
        """
        # set logging level
        set_log_level(log_level)
        self.model = None

    @staticmethod
    def _map_gpt4all_model_to_type(self, model_filename: str):
        # Map model filename to correct type.
        # Simple and most robust solution is to hard code this for now but
        # may need more flexible solution depending on how model list grows.

        gpt_j_models = [
            "ggml-gpt4all-j-v1.3-groovy.bin",
            "ggml-gpt4all-j-v1.2-jazzy.bin",
            "ggml-gpt4all-j-v1.1-breezy.bin",
            "ggml-gpt4all-j.bin",
        ]
        llama_models = ["ggml-vicuna-7b-1.1-q4_2.bin", "ggml-vicuna-13b-1.1-q4_2.bin"]

        if model_filename in gpt_j_models:
            return ModelType.GPT_J
        elif model_filename in llama_models:
            return ModelType.LLAMA
        else:
            return ModelType.UNDEFINED

    @staticmethod
    def list_models():
        response = requests.get("https://gpt4all.io/models/models.json")
        model_json = json.loads(response.content)
        return model_json

    def retrieve_model(self, model_filename: str = None, download_dir: str = ""):
        def get_download_url(model_filename):
            return f"https://gpt4all.io/models/{model_filename}"

        model_list = self.list_models()

        # If no provided filename, retrieve default model
        if model_filename is None:
            for item in model_list:
                if "isDefault" in item and item["isDefault"] == "true":
                    model_filename = item["filename"]
        else:
            # This will probably be a common error so let's try catching it
            if ".bin" not in model_filename:
                model_filename += ".bin"

        # Before attempting download, check that model is valid
        model_match = False
        for item in model_list:
            if model_filename == item["filename"]:
                model_match = True
                break
        if not model_match:
            raise ValueError("Invalid model filename")

        # Download model
        download_url = get_download_url(model_filename)
        download_path = None
        if os.path.exists(download_dir):
            download_path = os.path.join(download_dir, model_filename)
        else:
            raise ValueError("Invalid download directory")

        response = requests.get(download_url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1048576  # 1 MB
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(download_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        # Validate download was successful
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError(
                "An error occurred during download. Downloaded file may not work."
            )

        print("Model downloaded at: " + download_path)
        return download_path

    def load_model(
        self,
        model: str = None,
        allow_download: bool = True,
        model_type: ModelType = None,
        **model_params,
    ):
        # Take model path to load. If not exists, try retrieving model
        # This method will also need to handle llama or gptj
        # Allow for custom models

        model_filename = os.path.split(model)[-1]

        if os.is_file(model):
            pass
        elif allow_download:
            model_filepath = self.retrieve_model(model)

            pass

        return None

    def generate(
        self, prompt, max_token_limit=128, new_text_callback=None, **generate_params
    ):
        pass


if __name__ == "__main__":
    gpt4all = GPT4All()
    print(gpt4all.list_models())

    gpt4all.retrieve_model()
