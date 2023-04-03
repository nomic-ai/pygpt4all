# PyLLaMaCpp
Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) + A simple web UI

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pyllamacpp)](https://pypi.org/project/pyllamacpp/)

[//]: # ([![Wheels]&#40;https://github.com/abdeladim-s/pyllamacpp/actions/workflows/wheels.yml/badge.svg?branch=main&event=push&#41;]&#40;https://github.com/abdeladim-s/pyllamacpp/actions/workflows/wheels.yml&#41;)

[//]: # ([![Wheels-windows-mac]&#40;https://github.com/abdeladim-s/pyllamacpp/actions/workflows/wheels-windows_mac.yml/badge.svg&#41;]&#40;https://github.com/abdeladim-s/pyllamacpp/actions/workflows/wheels-windows_mac.yml&#41;)

[//]: # (<br/>)

[//]: # (<p align="center">)

[//]: # (  <img src="https://github.com/abdeladim-s/pyllamacpp/blob/main/docs/demo.gif?raw=true">)

[//]: # (</p>)


For those who don't know, `llama.cpp` is a port of Facebook's LLaMA model in pure C/C++:

<blockquote>

- Without dependencies
- Apple silicon first-class citizen - optimized via ARM NEON
- AVX2 support for x86 architectures
- Mixed F16 / F32 precision
- 4-bit quantization support
- Runs on the CPU

</blockquote>

# Table of contents
<!-- TOC -->
* [Installation](#installation)
* [Usage](#usage)
    * [Web UI](#web-ui)
    * [Python bindings](#python-bindings)
* [Discussions and contributions](#discussions-and-contributions)
* [License](#license)
<!-- TOC -->

# Installation
1. The easy way is to use the prebuilt wheels
```bash
pip install pyllamacpp
```

However, the compilation process of `llama.cpp` is taking into account the architecture of the target `CPU`, 
so you might need to build it from source:

```shell
git clone --recursive https://github.com/abdeladim-s/pyllamacpp && cd pyllamacpp
pip install .
```

# Usage

### Web UI
The package contains a simple web UI to test the bindings:

- Lightweight, and easy to use.
- Only needs Python.
- Has the option to convert the models to `ggml` format.
- A code like editor.
- Different options to tweak the `llama.cpp` parameters.
- Ability to export the generated text.

From the command line, run:
```shell
pyllamacpp-webui
```

That's it!<br>
A web page will be opened on your default browser, otherwise navigate to the links provided by the command.


### Python bindings

A simple `Pythonic` API is built on top of `llama.cpp` C/C++ functions. You can call it from Python as follows:

```python
from pyllamacpp.model import Model

def new_text_callback(text: str):
    print(text, end="")

model = Model(ggml_model='./models/ggml-model-f16-q4_0.bin', n_ctx=512, n_threads=8)
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
```
If you don't want to use the `callback`, you can get the results from the `generate` method once the inference is finished:

```python
generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)
```
* You can pass any `llama context` [parameter](https://abdeladim-s.github.io/pyllamacpp/#pyllamacpp.constants.LLAMA_CONTEXT_PARAMS_SCHEMA) as a keyword argument to the `Model` class
* You can pass any `gpt` [parameter](https://abdeladim-s.github.io/pyllamacpp/#pyllamacpp.constants.LLAMA_CONTEXT_PARAMS_SCHEMA) as a keyword argument to the `generarte` method
* You can always refer to the [short documentation](https://abdeladim-s.github.io/pyllamacpp/) for more details.

You can convert and quantize the models from Python as well:

# Supported model

### GPT4All
1. First [Get](https://github.com/nomic-ai/gpt4all#try-it-yourself) the gpt4all model.
2. Convert it to the new `ggml` format

On your terminal run: 
```shell
pyllamacpp-convert-gpt4all path/to/gpt4all_model.bin path/to/llama_tokenizer path/to/gpt4all-converted.bin
```

# Discussions and contributions
If you find any bug, please open an [issue](https://github.com/abdeladim-s/pyllamacpp/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/abdeladim-s/pyllamacpp/discussions) and open a new topic.

# License

This project is licensed under the same license as [llama.cpp](https://github.com/ggerganov/whisper.cpp/blob/master/LICENSE) (MIT  [License](./LICENSE)).

