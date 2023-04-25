#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions
"""
import logging
from pathlib import Path
import _pyllamacpp as pp
import pyllamacpp._logger
import sys
import json
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

def llama_to_ggml(dir_model: str, ftype: int = 1) -> str:
    """
    A helper function to convert LLaMa Pytorch models to ggml,
    same exact script as `convert-pth-to-ggml.py` from [llama.cpp](https://github.com/ggerganov/llama.cpp)
    repository, copied here for convinience purposes only!

    :param dir_model: llama model directory
    :param ftype: 0 or 1, 0-> f32, 1-> f16
    :return: ggml model path
    """
    # output in the same directory as the model
    assert ftype in [0, 1], f"ftype should be in [0,1], 0-> f32, 1-> f16"

    fname_hparams = str((Path(dir_model) / "params.json").absolute())
    fname_tokenizer = str((Path(dir_model).parent / "tokenizer.model").absolute())

    def get_n_parts(dim):
        if dim == 4096:
            return 1
        elif dim == 5120:
            return 2
        elif dim == 6656:
            return 4
        elif dim == 8192:
            return 8
        else:
            print("Invalid dim: " + str(dim))
            sys.exit(1)

    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    #
    # map from ftype to string
    ftype_str = ["f32", "f16"]

    with open(fname_hparams, "r") as f:
        hparams = json.load(f)

    tokenizer = SentencePieceProcessor(fname_tokenizer)

    hparams.update({"vocab_size": tokenizer.vocab_size()})

    n_parts = get_n_parts(hparams["dim"])

    print(hparams)
    print('n_parts = ', n_parts)

    for p in range(n_parts):
        print('Processing part ', p)

        # fname_model = dir_model + "/consolidated.00.pth"

        fname_model = str(Path(dir_model) / f"consolidated.0{str(p)}.pth")
        fname_out = str(Path(dir_model) / f"ggml-model-{ftype_str[ftype]}.bin")
        if (p > 0):
            fname_out = str(Path(dir_model) / f"ggml-model-{ftype_str[ftype]}.bin.{str(p)}")

        model = torch.load(fname_model, map_location="cpu")

        fout = open(fname_out, "wb")

        fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
        fout.write(struct.pack("i", hparams["vocab_size"]))
        fout.write(struct.pack("i", hparams["dim"]))
        fout.write(struct.pack("i", hparams["multiple_of"]))
        fout.write(struct.pack("i", hparams["n_heads"]))
        fout.write(struct.pack("i", hparams["n_layers"]))
        fout.write(struct.pack("i", hparams["dim"] // hparams["n_heads"]))  # rot (obsolete)
        fout.write(struct.pack("i", ftype))

        # Is this correct??
        for i in range(32000):
            if tokenizer.is_unknown(i):
                # "<unk>" token (translated as ??)
                text = " \u2047 ".encode("utf-8")
                fout.write(struct.pack("i", len(text)))
                fout.write(text)
            elif tokenizer.is_control(i):
                # "<s>"/"</s>" tokens
                fout.write(struct.pack("i", 0))
            elif tokenizer.is_byte(i):
                # "<U+XX>" tokens (which may be invalid UTF-8)
                piece = tokenizer.id_to_piece(i)
                if len(piece) != 6:
                    print("Invalid token: " + piece)
                    sys.exit(1)
                byte_value = int(piece[3:-1], 16)
                fout.write(struct.pack("i", 1))
                fout.write(struct.pack("B", byte_value))
            else:
                # normal token. Uses U+2581 (LOWER ONE EIGHTH BLOCK) to represent spaces.
                text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
                fout.write(struct.pack("i", len(text)))
                fout.write(text)

        for k, v in model.items():
            name = k
            shape = v.shape

            # skip layers.X.attention.inner_attention.rope.freqs
            if name[-5:] == "freqs":
                continue

            print("Processing variable: " + name + " with shape: ", shape, " and type: ", v.dtype)

            # data = tf.train.load_variable(dir_model, name).squeeze()
            data = v.numpy().squeeze()
            n_dims = len(data.shape);

            # for efficiency - transpose some matrices
            # "model/h.*/attn/c_attn/w"
            # "model/h.*/attn/c_proj/w"
            # "model/h.*/mlp/c_fc/w"
            # "model/h.*/mlp/c_proj/w"
            # if name[-14:] == "/attn/c_attn/w" or \
            #   name[-14:] == "/attn/c_proj/w" or \
            #   name[-11:] == "/mlp/c_fc/w" or \
            #   name[-13:] == "/mlp/c_proj/w":
            #    print("  Transposing")
            #    data = data.transpose()

            dshape = data.shape

            # default type is fp16
            ftype_cur = 1
            if ftype == 0 or n_dims == 1:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

            # header
            sname = name.encode('utf-8')
            fout.write(struct.pack("iii", n_dims, len(sname), ftype_cur))
            for i in range(n_dims):
                fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
            fout.write(sname);

            # data
            data.tofile(fout)

        # I hope this deallocates the memory ..
        model = None

        fout.close()

        print("Done. Output file: " + fname_out + ", (part ", p, ")")
        print("")
        return fname_out


def quantize(ggml_model_path: str, output_model_path: str = None, itype: int = 2) -> str:
    """
    Qunatizes the ggml model.

    :param ggml_model_path: path of the ggml model
    :param output_model_path: output file path for the qunatized model
    :param itype: quantization type: 2 -> Q4_0, 3 -> Q4_1
    :return: quantized model path
    """
    if output_model_path is None:
        output_model_path = ggml_model_path + f'-q4_{0 if itype == 2 else 1}.bin'
    logging.info("Quantization will start soon ... (This my take a while)")
    pp.llama_quantize(ggml_model_path, output_model_path, itype)
    logging.info(f"Quantized model is created successfully {output_model_path}")
    return output_model_path


def convert_gpt4all() -> str:
    pass
