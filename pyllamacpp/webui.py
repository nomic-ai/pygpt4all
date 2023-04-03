#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyLLaMaCpp Web User Interface (web UI)
"""

import importlib.metadata
import sys
import streamlit as st
from streamlit import runtime
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
from streamlit.web import cli as stcli

__author__ = "abdeladim-s"
__github__ = "https://github.com/abdeladim-s/pyllamacpp"
__copyright__ = "Copyright 2023,"
__deprecated__ = False
__version__ = importlib.metadata.version('pyllamacpp')

from pyllamacpp.constants import GPT_PARAMS_SCHEMA
from pyllamacpp.model import Model
import pyllamacpp.utils as utils


# from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx, RerunException

def _get_key(model_name: str, config_name: str) -> str:
    """
    a simple helper method to generate unique key for configs UI

    :param model_name: name of the model
    :param config_name: configuration key
    :return: str key
    """
    return model_name + '-' + config_name


def _config_ui(config_name: str, key: str, config: dict):
    """
    helper func that returns the config UI based on the type of the config

    :param config_name: the name of the model
    :param key: the key to set for the config ui
    :param config: configuration object

    :return: config UI streamlit objects
    """
    if config['type'] == str:
        return st.text_input(config_name, help=config['description'], key=key, value=config['default'])
    elif config['type'] == list:
        return st.selectbox(config_name, config['options'], index=config['options'].index(config['default']),
                            help=config['description'], key=key)
    elif config['type'] == float or config['type'] == int:
        if config['default'] is None:
            return st.text_input(config_name, help=config['description'], key=key, value=config['default'])
        return st.number_input(label=config_name, help=config['description'], key=key, value=config['default'])
    elif config['type'] == bool:
        return st.checkbox(label=config_name, value=config['default'], help=config['description'], key=key)
    else:
        print(f'Warning: {config_name} does not have a supported UI')
        pass


def _generate_config_ui(model_name, config_schema):
    """
    Loops through configuration dict object and generates the configuration UIs
    :param model_name:
    :param config_schema:
    :return: Config UIs
    """
    for config_name in config_schema:
        config = config_schema[config_name]
        key = _get_key(model_name, config_name)
        _config_ui(config_name, key, config)


def _get_config_from_session_state(model_name: str, config_schema: dict, notification_placeholder=None) -> dict:
    """
    Helper function to get configuration dict from the generated config UIs

    :param model_name: name of the model
    :param config_schema: configuration schema
    :param notification_placeholder: notification placeholder streamlit object in case of errors

    :return: dict of configs
    """
    model_config = {}
    for config_name in config_schema:
        key = _get_key(model_name, config_name)
        try:
            value = st.session_state[key]
            if config_schema[config_name]['type'] == str:
                if value == 'None' or value == '':
                    value = None
            elif config_schema[config_name]['type'] == float:
                if value == 'None' or value == '':
                    value = None
                else:
                    value = float(value)
            elif config_schema[config_name]['type'] == int:
                if value == 'None' or value == '':
                    value = None
                else:
                    value = int(value)

            model_config[config_name] = value
        except KeyError as e:
            pass
        except Exception as e:
            if notification_placeholder:
                notification_placeholder.error(f'Problem parsing configs!! \n {e}')
            return
    return model_config


@st.cache_resource
def _create_model(ggml_model: str, n_ctx: int):
    """
    Returns a llama.cpp model

    :param ggml_model: ggml model path
    :param n_ctx: n_ctx

    :return: llama.cpp model
    """
    model = Model(ggml_model=ggml_model, n_ctx=n_ctx)
    return model


footer = """
<style>
    #page-container {
      position: relative;
    }

    footer{
        visibility:hidden;
    }

    .footer {
    position: relative;
    left: 0;
    top:230px;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #808080; /* theme's text color hex code at 50 percent brightness*/
    text-align: left; /* you can replace 'left' with 'center' or 'right' if you want*/
    }
</style>

<div id="page-container">
    <div class="footer">
        <p style='font-size: 0.875em;'>
        Made with ❤ by <a style='display: inline; text-align: left;' href="https://github.com/abdeladim-s" target="_blank">abdeladim-s</a></p>
    </div>
</div>
"""


def _init_session_state(key: str, val: any):
    if key not in st.session_state:
        st.session_state[key] = val


# init session state
_init_session_state('generated_text', "")
_init_session_state('generate_button_disabled', True)
_init_session_state('model', None)
_init_session_state('prompt', "")
_init_session_state('ggml_model_path', "")


def new_text_generated(text):
    print(text)
    st.session_state['generated_text'] += text


def webui() -> None:
    """
    main web UI
    :return: None
    """
    st.set_page_config(page_title='PyLLaMaCpp',
                       page_icon=":llama:",
                       menu_items={
                           'Get Help': 'https://github.com/abdeladim-s/pyllamacpp',
                           'Report a bug': "https://github.com/abdeladim-s/pyllamacpp/issues",
                           'About': f"### [PyLLaMaCpp](https://github.com/abdeladim-s/pyllamacpp) \nv{__version__} "
                                    f"\n \nLicense: MIT"
                       },
                       layout="wide",
                       initial_sidebar_state='auto')

    st.markdown(f"# PyLLaMaCpp :llama:")
    st.markdown(
        "#### A simple Web UI for [llama.cpp](https://github.com/ggerganov/llama.cpp) Python [bindings]("
        "https://github.com/abdeladim-s/pyllamacpp)")

    notification_placeholder = st.empty()

    with st.sidebar:
        st.title("Settings")
        with st.expander('Model Conversion', expanded=False):
            st.info("* This section is intended to help you convert an original LLaMa model to a ggml model.\n"
                    "* If you have a ggml model (or you have already did the conversion before), load it directly in "
                    "the next section\n")
            llama_model_dir = st.text_input('LLaMa Model directory', help='Absolute path of the LLaMa model directory')
            quantize = st.checkbox("Quantize", help="Quantization reduces the model size while keeping it accurate",
                                   value=True)
            # if quantize:
            #     quantization_type = st.selectbox("Qnatization type", options=['Q4_0', 'Q4_1'])
            convert_button = st.button("Convert")
            if convert_button:
                with st.spinner("Processing (This may take a while) ..."):
                    st.session_state['ggml_model_path'] = utils.llama_to_ggml(llama_model_dir)
                    if quantize:
                        st.session_state['ggml_model_path'] = utils.quantize(st.session_state['ggml_model_path'])

        with st.expander('Load Model', expanded=True):
            ggml_model_path = st.text_input("ggml model path", value=st.session_state['ggml_model_path'])
            n_ctx = st.number_input("n_ctx", help="The maximum number of tokens of the prompt", value=512)
            load_button = st.button("Load")
            if load_button:
                st.session_state['model'] = _create_model(ggml_model_path, n_ctx)
                st.session_state['generate_button_disabled'] = False

        with st.expander('Generation params', expanded=False):
            _generate_config_ui("", GPT_PARAMS_SCHEMA)

        generate_button = st.button('Generate', type='primary', disabled=st.session_state['generate_button_disabled'],
                                    help="Please load the model first" if st.session_state[
                                        'generate_button_disabled'] else "Generate new text")

    if generate_button:
        params = _get_config_from_session_state("", GPT_PARAMS_SCHEMA, notification_placeholder)
        st.session_state['model'].generate(st.session_state['prompt'], new_text_callback=new_text_generated,
                                           verbose=True, **params)

    with st.expander("Editor configs"):
        language = st.selectbox("Language mode", options=LANGUAGES, index=113)  # 121 python, 113 plain_text
        font_size = st.slider("Font size", 5, 24, 18)
        theme = st.selectbox("Theme", options=THEMES, index=26),
        keybinding = st.selectbox("Keybinding mode", options=KEYBINDINGS, index=3)
        wrap = st.checkbox("Wrap enabled", value=True)
        show_gutter = st.checkbox("Show gutter", value=True)
        show_print_margin = st.checkbox("Show print margin", value=False)
        auto_update = st.checkbox("Auto update", value=False)
    content = st_ace(
        placeholder="Write your prompt here",
        language=language,
        theme=theme,
        keybinding=keybinding,
        font_size=font_size,
        show_gutter=show_gutter,
        show_print_margin=show_print_margin,
        wrap=wrap,
        auto_update=auto_update,
        min_lines=15,
        value=st.session_state['prompt'] + st.session_state['generated_text']
    )
    if content:
        st.session_state['prompt'] = content
        st.session_state['generated_text'] = ""

    with st.expander("Export file"):
        file_name = st.text_input("file name", value='llama.txt')
        st.download_button('Download', content, file_name=file_name)

    st.info(
        "This is an open source project. You are welcome to **contribute** your awesome "
        "comments, questions, ideas through "
        "[discussions](https://github.com/abdeladim-s/pyllamacpp/discussions), "
        "[issues](https://github.com/abdeladim-s/pyllamacpp/issues) and "
        "[pull requests](https://github.com/abdeladim-s/pyllamacpp/pulls) "
        "to the [project repository](https://github.com/abdeladim-s/pyllamacpp/). "
    )
    st.markdown(footer, unsafe_allow_html=True)


def run():
    if runtime.exists():
        webui()
    else:
        sys.argv = ["streamlit", "run", __file__, "--theme.base", "light"]
        sys.exit(stcli.main())


if __name__ == '__main__':
    run()
