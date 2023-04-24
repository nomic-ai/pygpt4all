#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _pygptj as pp

model_path = "/home/su/Downloads/ggml-gpt4all-j.bin"

def new_text_callback(text: str):
    print(text, end='')

def main():
    # loading the model
    model = pp.gptj_model()
    vocab = pp.gpt_vocab()
    pp.gptj_model_load(model_path, model, vocab)
    # gnerate
    params = pp.gptj_gpt_params()
    params.prompt = "Hi, "
    params.n_predict = 55
    # tokens = pp.gpt_tokenize(vocab, "hey you!")
    # print(tokens)
    # pp.gptj_free(model)
    pp.gptj_generate(params, model, vocab, new_text_callback)


if __name__ == '__main__':
    main()
