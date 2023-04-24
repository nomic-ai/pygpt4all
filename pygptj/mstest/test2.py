#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pygptj.model import Model

def new_text_callback(text):
    print(text, end="")

model = Model('/home/su/Downloads/ggml-gpt4all-j.bin')
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)

if __name__ == '__main__':
    pass
