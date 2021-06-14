#!/usr/bin/env python
"""
Usage:
    monolingualprocess.py bert-tokenize [options] INPUT_DATA OUTPUT_DATA_PATH
    monolingualprocess.py bert-tokenize multiple [options] INPUT_DATA_LIST OUTPUT_DATA_PATH

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --vocab-size=<num>         The vocabulary size. [default: 25000]
    --pct-bpe=<pct>            Percent of the vocabulary size to be BPE. [default: 0.1]
    --bert-model=<model>       Pretrained BERT model to use from pytorch-transformers. [default: bert-base-cased]
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from itertools import chain
from tqdm import tqdm

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath


def load_data(path: str):
    with open(path) as f:
        for line in f:
            yield line.strip()

def run(arguments):

    if arguments['multiple']:
        all_data = []
        for input_file in arguments['INPUT_DATA_LIST'].split(','):
            all_data.append(load_data(input_file))
        data = chain(*all_data)
    else:
        in_data_path = arguments['INPUT_DATA']
        data = load_data(in_data_path)

    out_data_path = RichPath.create(arguments['OUTPUT_DATA_PATH'])


    from pytorch_transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(arguments['--bert-model'])

    logging.info('Converting data...')
    def bpe_convertor():
        for line in tqdm(data):
            tokens = tokenizer.tokenize(line)
            if len(tokens) < 4:
                continue
            yield tokens
    out_data_path.save_as_compressed_file(bpe_convertor())


if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
