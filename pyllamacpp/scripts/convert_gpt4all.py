import argparse

from pyllamacpp.scripts.convert import convert_one_file
from pyllamacpp.scripts.migrate import main as migrate

from sentencepiece import SentencePieceProcessor

def main():
    parser = argparse.ArgumentParser(description='Convert GPT4All model to the current ggml format')
    parser.add_argument('gpt4all_model', help='path to gpt4all-lora-quantized.bin')
    parser.add_argument('tokenizer_model', help='path to LLaMA tokenizer.model file')
    parser.add_argument('fout_path', help='your new ggjt file name')

    args = parser.parse_args()

    print(args)

    tokenizer = SentencePieceProcessor(args.tokenizer_model)
    convert_one_file(args.gpt4all_model, tokenizer)
    from collections import namedtuple

    Files = namedtuple('Files', 'fin_path fout_path')
    files = Files(args.gpt4all_model, args.fout_path)
    migrate(files)


if __name__ == '__main__':
    main()