#!/usr/bin/env python
"""
Usage:
    outputparallelpredictions.py [options] MODEL_FILENAME TEST_DATA OUT_PREFIX

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --data-type=<type>         The type of data to be used. Possible options fce, code, wikiatomicedits, wikiedits. [default: fce]
    --greedy                   Use greedy decoding rather than beam search.
    --cpu                      Use cpu only.
    --verbose                  Print predictions to console.
    --num-predictions=N        Number of predictions to output. [default: 1]
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from typing import List

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

from data.loading import load_data_by_type
from dpu_utils.ptutils import BaseComponent

def __join_sentence(token_list: List[str]) -> str:
    s = ''
    in_bpe = False
    for t in token_list:
        if t == '__sow':
            in_bpe = True
        elif t == '__eow':
            in_bpe = False
            s += ' '
        elif in_bpe:
            s += t
        elif t.startswith('##'):
            s = s[:-1] + t[2:] + ' '
        else:
            s += t + ' '
    return s

def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)
    num_predictions = int(arguments['--num-predictions'])

    model_path = RichPath.create(arguments['MODEL_FILENAME'], azure_info_path)

    if arguments['--cpu']:
        model = BaseComponent.restore_model(model_path, 'cpu')
    else:
        model = BaseComponent.restore_model(model_path)

    test_data_path = RichPath.create(arguments['TEST_DATA'], azure_info_path)
    test_data = load_data_by_type(test_data_path, arguments['--data-type'])

    logging.info('Running test on %d examples', len(test_data))

    model.eval()
    all_data = [model.load_data_from_sample(d) for d in test_data]
    ground_input_sequences = [d.input_sequence for d in test_data]

    data_iter = iter(all_data)
    predictions = []
    is_full = True

    start_idx = 0
    while is_full:
        mb_data, is_full, num_elements = model.create_minibatch(data_iter, max_num_items=10)
        mb_ground_input_sequences= ground_input_sequences[start_idx:start_idx + num_elements]
        if num_elements > 0:
            logging.info('Before decoding predictions: %d', len(predictions))
            if arguments['--greedy']:
                predicted_outputs = model.greedy_decode(input_sequences=mb_data['input_sequences'],
                                                    ground_input_sequences=mb_ground_input_sequences)
            else:
                predicted_outputs = model.beam_decode(input_sequences=mb_data['input_sequences'],
                                                    ground_input_sequences=mb_ground_input_sequences)
            predictions.extend(predicted_outputs)
            logging.info('After decoding predictions: %d', len(predictions))

            start_idx += num_elements
        if not is_full:
            break

    prediction_files = []
    for i in range(num_predictions):
        prediction_files.append(open(arguments['OUT_PREFIX'] + f'-after-{i}.txt', 'w'))
    with open(arguments['OUT_PREFIX'] + '-before.txt', 'w') as before_f:
        assert len(ground_input_sequences) == len(predictions)
        for ground, predicted_beam in zip(ground_input_sequences, predictions):
            before_f.write(__join_sentence(ground) + '\n')
            for i in range(num_predictions):
                predicted_sentence = predicted_beam[0][i]
                predicted_sentence = __join_sentence(predicted_sentence)
                prediction_files[i].write(predicted_sentence + '\n')


if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
