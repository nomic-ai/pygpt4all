# PyLLaMACpp
Official supported Python bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp) + gpt4all

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pyllamacpp)](https://pypi.org/project/pyllamacpp/)


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
* [Supported model](#supported-model)
    * [GPT4All](#gpt4all)
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
git clone --recursive https://github.com/nomic-ai/pyllamacpp && cd pyllamacpp
pip install .
```

# Usage

A simple `Pythonic` API is built on top of `llama.cpp` C/C++ functions. You can call it from Python as follows:

```python
from pyllamacpp.model import Model

def new_text_callback(text: str):
    print(text, end="", flush=True)

model = Model(ggml_model='./models/gpt4all-model.bin', n_ctx=512)
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback, n_threads=8)
```
If you don't want to use the `callback`, you can get the results from the `generate` method once the inference is finished:

```python
generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)
```

## Interactive Mode

If you want to run the program in interactive mode you can add the `grab_text_callback` function and set `interactive` to True in the generate function. `grab_text_callback` should always return a string unless you wish to signal EOF in which case you should return None.

```py
from pyllamacpp.model import Model

def new_text_callback(text: str):
    print(text, end="", flush=True)

def grab_text_callback():
    inpt = input()
    # To signal EOF, return None
    if inpt == "END":
        return None
    return inpt

model = Model(ggml_model='./models/gpt4all-model.bin', n_ctx=512)

# prompt from https://github.com/ggerganov/llama.cpp/blob/master/prompts/chat-with-bob.txt
prompt = """
Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision. To do this, Bob uses a database of information collected from many different sources, including books, journals, online articles, and more.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User:"""

model.generate(prompt, n_predict=256, new_text_callback=new_text_callback, grab_text_callback=grab_text_callback, interactive=True, repeat_penalty=1.0, antiprompt=["User:"])
```

* You can pass any `llama context` [parameter](https://nomic-ai.github.io/pyllamacpp/#pyllamacpp.constants.LLAMA_CONTEXT_PARAMS_SCHEMA) as a keyword argument to the `Model` class
* You can pass any `gpt` [parameter](https://nomic-ai.github.io/pyllamacpp/#pyllamacpp.constants.GPT_PARAMS_SCHEMA) as a keyword argument to the `generarte` method
* You can always refer to the [short documentation](https://nomic-ai.github.io/pyllamacpp/) for more details.


# Supported model

### GPT4All

Download a GPT4All model from https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/.
The easiest approach is download a file whose name ends in `ggml.bin`--older model versions require conversion.

If you have an older model downloaded that you want to convert, in your terminal run: 
```shell
pyllamacpp-convert-gpt4all path/to/gpt4all_model.bin path/to/llama_tokenizer path/to/gpt4all-converted.bin
```

# FAQs
* Where to find the llama tokenizer? [#5](https://github.com/nomic-ai/pyllamacpp/issues/5)

# Discussions and contributions
If you find any bug, please open an [issue](https://github.com/nomic-ai/pyllamacpp/issues).

If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions](https://github.com/nomic-ai/pyllamacpp/discussions) and open a new topic.

# License

This project is licensed under the same license as [llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE) (MIT  [License](./LICENSE)).

