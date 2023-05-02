# PyGPT4All
Official Python CPU inference for [GPT4All](https://github.com/nomic-ai/gpt4all) language models based on [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pygpt4all)](https://pypi.org/project/pygpt4all/)

<!-- TOC -->
* [Installation](#installation)
* [Tutorial](#tutorial)
    * [Model instantiation](#model-instantiation)
    * [Simple generation](#simple-generation)
    * [Interactive Dialogue](#interactive-dialogue)
* [API reference](#api-reference)
* [License](#license)
<!-- TOC -->
# Installation

```bash
pip install pygpt4all
```

# Tutorial

You will need first to download the model weights

| Model     | Download link                                            |
|-----------|----------------------------------------------------------|
| GPT4ALL   | http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin    |
| GPT4ALL-j | https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin |                                                                     

### Model instantiation
Once the weights are downloaded, you can instantiate the models as follows:
* GPT4All model

```python
from pygpt4all import GPT4All

model = GPT4All('path/to/ggml-gpt4all-l13b-snoozy.bin')
```

* GPT4All-J model

```python
from pygpt4all import GPT4All_J

model = GPT4All_J('path/to/ggml-gpt4all-j-v1.3-groovy.bin')
```


### Simple generation
The `generate` function is used to generate new tokens from the `prompt` given as input:

```python
for token in model.generate("Tell me a joke ?\n"):
    print(token, end='', flush=True)
```

### Interactive Dialogue
You can set up an interactive dialogue by simply keeping the `model` variable alive:

```python
while True:
    try:
        prompt = input("You: ", flush=True)
        if prompt == '':
            continue
        print(f"AI:", end='')
        for token in model.generate(prompt):
            print(f"{token}", end='', flush=True)
        print()
    except KeyboardInterrupt:
        break
```

# API reference
You can check the [API reference documentation](https://nomic-ai.github.io/pygpt4all/) for more details.


# License
This project is licensed under the MIT  [License](./LICENSE).

