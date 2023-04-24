# PyGPT-J
Official supported Python bindings for [GPT4All-J](https://github.com/nomic-ai/gpt4all#raw-model) language model based on [ggml](https://github.com/ggerganov/ggml).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[//]: # ([![PyPi version]&#40;https://badgen.net/pypi/v/pyllamacpp&#41;]&#40;https://pypi.org/project/pyllamacpp/&#41;)

# Table of contents
<!-- TOC -->
* [Installation](#installation)
* [Usage](#usage)
* [GPT4All-J](#gpt4all-j)
* [License](#license)
<!-- TOC -->

# Installation
1. The easy way is to use the prebuilt wheels
```bash
pip install pygptj
```

2. Build it from source:

```shell
git clone --recursive https://github.com/abdeladim-s/pygptj && cd pygptj
pip install .
```

# Usage

```python
from pygptj.model import Model

 def new_text_callback(text):
        print(text, end="")

model = Model('./models/ggml-gpt4all-j.bin')
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
```
If you don't want to use the `callback`, you can get the results from the `generate` method once the inference is finished:

```python
generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)
```

[//]: # (* You can always refer to the [short documentation]&#40;https://nomic-ai.github.io/pyllamacpp/&#41; for more details.)


# GPT4All-J Model

Download the [GPT4All-J model](https://gpt4all.io/models/ggml-gpt4all-j.bin).

[//]: # (# Discussions and contributions)

[//]: # (If you find any bug, please open an [issue]&#40;https://github.com/nomic-ai/pyllamacpp/issues&#41;.)

[//]: # ()
[//]: # (If you have any feedback, or you want to share how you are using this project, feel free to use the [Discussions]&#40;https://github.com/nomic-ai/pyllamacpp/discussions&#41; and open a new topic.)

# License

This project is licensed under the MIT  [License](./LICENSE).

