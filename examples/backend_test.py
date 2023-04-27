######
# Project       : GPT4ALL-UI
# File          : backend_test.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# Licence       : Apache 2.0
# Description   : 
# This is an example of a pygpt4all-ui binding for llamacpp and gpt-j models
# Tests generation

# To call :
# python backend_test.py -m <model path> --prompt <your prompt> --trigger_stop_after <trigger stop after how many tokens? to test how the backend handles sopping generation>
######
from pathlib import Path
from typing import Callable
from pyllamacpp.model import Model
import argparse
import sys

__author__ = "parisneo"
__github__ = "https://github.com/nomic-ai/gpt4all-ui"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

backend_name = "LLAMACPP"

class LLAMACPP():
    file_extension='*.bin'
    def __init__(self, model_path, config:dict) -> None:
        """Builds a LLAMACPP backend

        Args:
            config (dict): The configuration file
        """
        self.config = config
        
        self.model = Model(
                ggml_model=model_path, 
                n_ctx=self.config['ctx_size'], 
                seed=self.config['seed'],
                )

    def get_num_tokens(self, prompt):
        return self.model.num_tokens(prompt)
    
    def generate(self, 
                 prompt:str,                  
                 n_predict: int = 128,
                 new_text_callback: Callable[[str], None] = bool,
                 verbose: bool = False,
                 **gpt_params ):
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            new_text_callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        try:
            self.model.generate(
                prompt,
                new_text_callback=new_text_callback,
                n_predict=n_predict,
                temp=self.config['temp'],
                top_k=self.config['top_k'],
                top_p=self.config['top_p'],
                repeat_penalty=self.config['repeat_penalty'],
                repeat_last_n = self.config['repeat_last_n'],
                n_threads=self.config['n_threads'],
                verbose=verbose
            )
        except Exception as ex:
            print(ex)

if __name__=="__main__":
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # add the -m or --model_path argument (./models/llama_cpp/ is for gpt4all-ui default structure)
    parser.add_argument("-m", "--model_path", default="./models/llama_cpp/", help="path to the model file")
    parser.add_argument('--temp', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--repeat_penalty', type=float, default=1.3)
    parser.add_argument('--repeat_last_n', type=int, default=5)
    parser.add_argument('--n_threads', type=int, default=8)
    parser.add_argument('--ctx_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=-1) 
    parser.add_argument('--prompt', type=str, default='Once apon a time')   
    parser.add_argument('--trigger_stop_after', type=int, default=-1)   
    # parse the arguments
    args = parser.parse_args()
    config = {
    'temp': args.temp,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'repeat_penalty': args.repeat_penalty,
    'repeat_last_n': args.repeat_last_n,
    'n_threads': args.n_threads,
    'ctx_size': args.ctx_size,
    'seed': args.seed    
    }
    
    backend = LLAMACPP(args.model_path, config)

    #Not good to use global, but this is a quick example so nevermind
    global counter
    counter = 0

    def callback(text):
        global counter
        print(text,end="")
        sys.stdout.flush()
        # test the stop generation after a number of words
        counter +=1        
        if args.trigger_stop_after>0:
            if counter>=args.trigger_stop_after:
                return False
            
        return True
    
    num_tokens = backend.get_num_tokens(args.prompt)
    print(f"Prompt has {num_tokens} tokens")
    output_text = backend.generate(args.prompt,new_text_callback=callback)
    print("Text : output_text")
