#!/usr/bin/env python
"""
Usage:
    convertcnndmgraphs.py INPUTS_JSONL SUMMARIES_JSONL OUTPUT_DATA_PATH

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
from typing import Iterator, Dict, Union

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

from data.edits import Edit, NLEdit

def parse_jsonl_edit_data(inputs_path: RichPath, target_path: RichPath) -> Iterator[Edit]:
    for i, (input_graph, target_sequence) in enumerate(zip(inputs_path.read_as_jsonl(), target_path.read_as_jsonl())):
        input_sequence = [input_graph['node_labels'][idx] for idx in input_graph['backbone_sequence']]
        yield Edit(
            input_sequence=input_sequence,
            output_sequence=target_sequence,
            provenance='L'+str(i),
            edit_type=''
        )

def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)

    inputs_path = RichPath.create(arguments['INPUTS_JSONL'], azure_info_path)
    summaries_path = RichPath.create(arguments['SUMMARIES_JSONL'], azure_info_path)
    out_data_path = RichPath.create(arguments['OUTPUT_DATA_PATH'], azure_info_path)

    data = parse_jsonl_edit_data(inputs_path, summaries_path)

    def to_dict():
        for edit in data:
            yield edit._asdict()

    out_data_path.save_as_compressed_file(to_dict())

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get('--debug', False))
