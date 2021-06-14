#!/usr/bin/env python
"""
Usage:
    score.py [options] MODEL_FILENAME

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --cpu                      Use cpu only.
    --verbose                  Print predictions to console.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from typing import Tuple

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from data.edits import Edit

from dpu_utils.ptutils import BaseComponent


def score(model: BaseComponent, sample_edit: Edit) -> Tuple[float, float]:
    model.eval()
    ground_mb_data = [model.load_data_from_sample(sample_edit)]

    mb_data, _, _ = model.create_minibatch(ground_mb_data, max_num_items=1)
    log_likelihood = float(model.compute_likelihood(**mb_data))
    return log_likelihood, log_likelihood / len(sample_edit.output_sequence)


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    model_path = RichPath.create(arguments['MODEL_FILENAME'], azure_info_path)

    if arguments['--cpu']:
        model = BaseComponent.restore_model(model_path, 'cpu')
    else:
        model = BaseComponent.restore_model(model_path)

    sample_edit = Edit(
        input_sequence = ["var",  "VAR0", "=", "(", "Math", ".", "Abs",  "(", "VAR1", ".", "GetHashCode", "(", ")", ")", "%", "VAR2", ")", ";" ],
        output_sequence=[ "var", "VAR0",  "=",  "Math", ".",  "Abs",  "(",  "MurmurHash",  ".",  "StringHash",   "(",  "VAR1",  ")",  ")", "%", "VAR2", ";" ],
        provenance="",
        edit_type=[]
    )

    sample_logprob, sample_per_token_entropy = score(model, sample_edit)
    print(f'Log prob {sample_logprob:.2f}  Per token entropy: {sample_per_token_entropy:.2f}')

if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
