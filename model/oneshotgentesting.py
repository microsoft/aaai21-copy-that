#!/usr/bin/env python3
"""
Test the ability of the model to do one-shot generation, given an edit representation of a different sample of the same edit type.
Usage:
    oneshotgentesting.py [options] MODEL_FILENAME DATA

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --sample-per-type=<num>    Number of samples per type
    --data-type=<type>         The type of data to be used.
    --cpu                      Use cpu only.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from collections import defaultdict
from typing import List

import torch
from tqdm import tqdm
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from data.edits import Edit

from data.loading import load_data_by_type
from dpu_utils.ptutils import BaseComponent


def evaluate_oneshot(model: BaseComponent, evaluate_oneshot: List[Edit], limit_per_category: int):
    model.eval()
    logging.info('Tensorizing data...')
    all_data = [model.load_data_from_sample(d) for d in evaluate_oneshot]
    data_iter = iter(all_data)

    representations = []
    continue_iterating = True

    logging.info('Computing edit representation on %d examples', len(evaluate_oneshot))
    start_idx = 0
    while continue_iterating:
        mb_data, continue_iterating, num_elements = model.create_minibatch(data_iter, max_num_items=20)
        if num_elements > 0:
            representations.extend(model.get_edit_representations(mb_data))
        else:
            assert not continue_iterating
        start_idx += num_elements

    assert len(representations) == len(all_data)

    # Do an all-vs-all
    sample_idxs_by_type = defaultdict(list)
    for i, edit in enumerate(evaluate_oneshot):
        sample_idxs_by_type[edit.edit_type].append(i)

    num_samples_per_type = defaultdict(int)
    num_correct_per_type = defaultdict(int)
    num_correct_at5_per_type = defaultdict(int)
    for edit_type, sample_idxs in sample_idxs_by_type.items():
        samples_to_see = sample_idxs[:limit_per_category]
        for sample_idx in tqdm(samples_to_see, leave=False, dynamic_ncols=True, desc=edit_type):
            mb_samples = [all_data[i] for i in samples_to_see if i != sample_idx]
            sample_mb_data, continue_iterating, num_elements = model.create_minibatch(mb_samples,
                                                                                      max_num_items=len(mb_samples)+1)
            assert num_elements == len(samples_to_see) - 1 and not continue_iterating

            edit_representations = representations[sample_idx].unsqueeze(0).expand(len(samples_to_see)-1, -1)

            beam = model.beam_decode(input_sequences=sample_mb_data['input_sequences'],
                                      aligned_edits=None,
                                      ground_input_sequences=[evaluate_oneshot[i].input_sequence for i in samples_to_see if i != sample_idx],
                                      max_length=50,
                                      fixed_edit_representations=edit_representations)
            for i, other_sample_idx in enumerate((i for i in samples_to_see if i != sample_idx)):
                top_prediction = beam[i][0][0]
                if top_prediction == evaluate_oneshot[other_sample_idx].output_sequence:
                    num_correct_per_type[edit_type] += 1
                num_correct_at5_per_type[edit_type] += 1 if any(beam[i][0][k] == evaluate_oneshot[other_sample_idx].output_sequence for k in range(5)) else 0
                num_samples_per_type[edit_type] += 1
        print(f'\t{edit_type}\t Acc: {num_correct_per_type[edit_type]/num_samples_per_type[edit_type]:%} Acc@5: {num_correct_at5_per_type[edit_type]/num_samples_per_type[edit_type]:%}')


    total_correct, total_correct_at_5, total_elements = 0, 0, 0
    for edit_type, num_samples in num_samples_per_type.items():
        total_correct += num_correct_per_type[edit_type]
        total_correct_at_5 += num_correct_at5_per_type[edit_type]
        total_elements += num_samples
    print(f'Total: {total_correct/total_elements:%}')


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    model_path = RichPath.create(arguments['MODEL_FILENAME'], azure_info_path)

    if arguments['--cpu']:
        model = BaseComponent.restore_model(model_path, 'cpu')
    else:
        model = BaseComponent.restore_model(model_path)

    test_data_path = RichPath.create(arguments['DATA'], azure_info_path)
    test_data = load_data_by_type(test_data_path, arguments['--data-type'], cleanup=False, as_list=True)

    if arguments['--sample-per-type'] is None:
        lim = 100000
    else:
        lim = int(arguments['--sample-per-type'])
    evaluate_oneshot(model, test_data, limit_per_category=lim)


if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
