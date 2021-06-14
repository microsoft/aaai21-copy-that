#!/usr/bin/env python3
"""
Save the edit representations
Usage:
    exportrepresentations.py [options] MODEL_FILENAME DATA OUT_FILE

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --data-type=<type>         The type of data to be used. Possible options fce, code, wikiatomicedits, wikiedits. [default: fce]
    --cpu                      Use cpu only.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from typing import List

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from data.edits import Edit

from data.loading import load_data_by_type
from dpu_utils.ptutils import BaseComponent


def export(model: BaseComponent, test_data: List[Edit], output_path: str):
    model.eval()
    logging.info('Tensorizing data...')
    all_data = [model.load_data_from_sample(d) for d in test_data]
    data_iter = iter(all_data)

    representations = []
    continue_iterating = True

    logging.info('Computing edit representation on %d examples', len(test_data))
    start_idx = 0
    while continue_iterating:
        mb_data, continue_iterating, num_elements = model.create_minibatch(data_iter, max_num_items=200)
        if num_elements > 0:
            representations.extend(model.edit_encoder.get_summary(input_sequence_data=mb_data['aligned_edits']))
        else:
            assert not continue_iterating
        start_idx += num_elements

    assert len(representations) == len(all_data)

    np.savez_compressed(output_path, representations=representations)

def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    model_path = RichPath.create(arguments['MODEL_FILENAME'], azure_info_path)

    if arguments['--cpu']:
        model = BaseComponent.restore_model(model_path, 'cpu')
    else:
        model = BaseComponent.restore_model(model_path)

    test_data_path = RichPath.create(arguments['DATA'], azure_info_path)
    test_data = load_data_by_type(test_data_path, arguments['--data-type'], cleanup=False)


    export(model, test_data, arguments['OUT_FILE'])


if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
