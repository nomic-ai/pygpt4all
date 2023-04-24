# PyGPT4All
Official Python CPU inference for [GPT4All](https://github.com/nomic-ai/gpt4all) language models based on [llama.cpp](https://github.com/ggerganov/llama.cpp) and [ggml](https://github.com/ggerganov/ggml)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![PyPi version]&#40;https://badgen.net/pypi/v/ptgpt4all&#41;]&#40;https://pypi.org/project/pygpt4all/&#41;)

**NB: Under active development**

# Installation

```bash
pip install pygpt4all
```

# Usage

### GPT4All model

Download a GPT4All model from https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/

```python
from pygpt4all.models.gpt4all import GPT4All

def new_text_callback(text):
    print(text, end="")

model = GPT4All('./models/ggml-gpt4all-j.bin')
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
```

### GPT4All-J model

Download the GPT4All-J model from https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin

```python
from pygpt4all.models.gpt4all_j import GPT4All_J

def new_text_callback(text):
    print(text, end="")

model = GPT4All_J('./models/ggml-gpt4all-j.bin')
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
```

[//]: # (* You can always refer to the [short documentation]&#40;https://nomic-ai.github.io/pyllamacpp/&#41; for more details.)


# License



This project is licensed under the MIT  [License](./LICENSE).

