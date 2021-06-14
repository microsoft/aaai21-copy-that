#!/usr/bin/env python
"""
Usage:
    test.py [options] MODEL_FILENAME TEST_DATA

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --data-type=<type>         The type of data to be used. Possible options fce, code, wikiatomicedits, wikiedits. [default: fce]
    --no-prediction            Do not ask the model to make predictions.
    --test-size=<size>         Size of test set to use. Only need to specify if less than total.
    --greedy                   Use greedy decoding rather than beam search.
    --cpu                      Use cpu only.
    --verbose                  Print predictions to console.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from typing import List
from itertools import islice

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from data.diffviz import diff
from data.edits import Edit, NLEdit
from data.editevaluator import EditEvaluator

from data.loading import load_data_by_type
from data.nlrepresentationviz import NLRepresentationsVisualizer
from data.representationviz import RepresentationsVisualizer
from io import StringIO
from dpu_utils.ptutils import BaseComponent
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import os
import re
import sys
import tokenize
import torch

IDENTIFER = 'IDENTIFIER'
NUMBER = 'NUM'
STRING = 'STR'

SMOOTHING_FUNCTION = SmoothingFunction().method2

REF_FILE = 'ref.txt'
ORIG_FILE = 'orig.txt'
PRED_FILE = 'pred.txt'
GOLD_EDIT_M2 = 'gold_edit_m2'
PRED_EDIT_M2 = 'pred_edit_m2'

def test(model: BaseComponent, test_data: List[Edit], model_name: str,
    data_type: str, test_predictions: bool, greedy: bool, verbose: bool, mb_size=10):
    model.eval()
    all_data = [model.load_data_from_sample(d) for d in test_data]

    if 'context' in data_type and not model.get_hyperparameter('disable_context_copy'):
        ground_input_sequences = [d.input_sequence + d.context_sequence for d in test_data]
    else:
        ground_input_sequences = [d.input_sequence for d in test_data]

    data_iter = iter(all_data)
    predictions = []
    edit_type_predictions = []
    representations = []
    is_full = True
    gold_likelihood_values = []

    start_idx = 0
    while is_full:
        ground_mb_data = list(islice(data_iter, mb_size))
        is_full = len(ground_mb_data) == mb_size
        mb_data, _, num_elements = model.create_minibatch(ground_mb_data, max_num_items=mb_size)

        mb_ground_input_sequences= ground_input_sequences[start_idx:start_idx + num_elements]
        if num_elements > 0:
            if test_predictions:
                logging.info('Before decoding predictions: %d', len(predictions))

                if greedy:
                    if 'context' in data_type:
                        predicted_outputs = model.greedy_decode(input_sequences=mb_data['input_sequences'],
                            aligned_edits=mb_data['aligned_edits'],
                            context_sequences=mb_data['context_sequences'],
                            ground_input_sequences=mb_ground_input_sequences)
                    else:
                        predicted_outputs = model.greedy_decode(input_sequences=mb_data['input_sequences'],
                            aligned_edits=mb_data['aligned_edits'],
                            ground_input_sequences=mb_ground_input_sequences)

                else:
                    if 'context' in data_type:
                        predicted_outputs = model.beam_decode(input_sequences=mb_data['input_sequences'],
                                                            aligned_edits=mb_data['aligned_edits'],
                                                            context_sequences=mb_data['context_sequences'],
                                                            ground_input_sequences=mb_ground_input_sequences)
                    else:
                        predicted_outputs = model.beam_decode(input_sequences=mb_data['input_sequences'],
                                                            aligned_edits=mb_data['aligned_edits'],
                                                            ground_input_sequences=mb_ground_input_sequences)

                predictions.extend(predicted_outputs)
                logging.info('After decoding predictions: %d', len(predictions))

            representations.extend(model.edit_encoder.get_summary(input_sequence_data=mb_data['aligned_edits']))
            # gold_likelihood_values.extend(model.compute_likelihood(**mb_data))

            start_idx += num_elements
        if not is_full:
            break

    if test_predictions:
        assert len(all_data) == len(predictions)

    run_test_suite(predictions, test_data, representations, gold_likelihood_values,
        data_type, model_name, verbose)

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
        else:
            s += t + ' '
    return s

def visualize_representations(viz, test_data, representations, model_name):
    viz.nearest_neighbors_to_html(test_data, representations,
                                  datapoint_to_html=lambda d: diff(__join_sentence(d.input_sequence), __join_sentence(d.output_sequence)),
                                  outfile=model_name+'.html', num_neighbors=5, num_items_to_show=5000)

def run_test_suite(predictions, test_data, representations, gold_likelihood_values,
    data_type='code', model_name='', verbose=False):
    representations = np.array(representations)
    gold_likelihood_values = np.array(gold_likelihood_values)
    gold_probs = np.exp(gold_likelihood_values)

    if True: #'nl' not in data_type:
        visualize_representations(RepresentationsVisualizer(labeler=lambda d: d.edit_type.split('+')[0]),
                                  test_data, representations, model_name)
    else:
        visualize_representations(NLRepresentationsVisualizer(), test_data, representations, model_name)

    if verbose:
        for i, (datasample, candidates) in enumerate(zip(test_data, predictions)):
            print('Link: {}\n'.format(datasample.provenance))
            if 'nl' in data_type:
                print('NL Input: {}'.format(' '.join(datasample.nl_sequence)))

            print('\nCode Input: ')
            print(' '.join(datasample.input_sequence))
            print('\n\nGold (Log prob:{}, Prob:{}):'.format(gold_likelihood_values[i], gold_probs[i]))
            print(' '.join(datasample.output_sequence))

            for predicted_tokens, score in zip(*candidates):
                print('\nPredicted (Log prob:{}, Prob:{}):'.format(score, np.exp(score)))
                print(' '.join(predicted_tokens))
            print('--------------------------')
        sys.stdout.flush()

    exact_match_errors = 0
    structural_match_errors = 0

    input_copy_ranks = []
    gold_ranks = []
    orig_instances = []
    references = []
    selected_predictions = []
    edit_evaluator = EditEvaluator()

    for i, (datasample, (predicted_sequences, _)) in enumerate(zip(test_data, predictions)):
        if len(predicted_sequences) == 1:
            used_prediction_idx = 0
        else:
            for used_prediction_idx in range(len(predicted_sequences)):
                # Since we are testing on edits, predicting the input_sequence is always wrong. Thus skip it!
                if predicted_sequences[used_prediction_idx] != datasample.input_sequence:
                    break
            else:
                raise Exception('All output sequences are identical to input sequences. This cannot happen.')

        beam_strings = [' '.join(s) for s in predicted_sequences]
        gold_str = ' '.join(datasample.output_sequence)
        try:
           gold_ranks.append(beam_strings.index(gold_str))
        except ValueError:
            pass # not found

        input_str = ' '.join(datasample.input_sequence)
        try:
            input_copy_ranks.append(beam_strings.index(input_str))
        except ValueError:
            pass # not found

        if predicted_sequences[used_prediction_idx] != datasample.output_sequence:
            exact_match_errors += 1

        gold_token_types = get_token_types(datasample.output_sequence)
        predicted_token_types = get_token_types(predicted_sequences[used_prediction_idx])

        if gold_token_types != predicted_token_types:
            structural_match_errors += 1

        references.append([datasample.output_sequence])
        orig_instances.append(datasample.input_sequence)
        selected_predictions.append(predicted_sequences[used_prediction_idx])
        edit_evaluator.add_sample(datasample.input_sequence,
                                  datasample.output_sequence,
                                  predicted_sequences[used_prediction_idx])

    if len(selected_predictions) > 0:
        with open(model_name + '_' + REF_FILE, 'w+') as f:
            for r in references:
                f.write(' '.join(r[0]) + '\n')

        with open(model_name + '_' + ORIG_FILE, 'w+') as f:
            for o in orig_instances:
                f.write(' '.join(o) + '\n')

        with open(model_name + '_' + PRED_FILE, 'w+') as f:
            for s in selected_predictions:
                f.write(' '.join(s) + '\n')

        orig_sentence_bleu = compute_bleu(references, orig_instances)
        pred_sentence_bleu = compute_bleu(references, selected_predictions)

        if len(gold_ranks) == 0:
            avg_gold_rank = 0.0
        else:
            avg_gold_rank = sum(gold_ranks)/float(len(gold_ranks))

        if len(input_copy_ranks) == 0:
            avg_input_copy_rank = 0.0
        else:
            avg_input_copy_rank = sum(input_copy_ranks)/float(len(input_copy_ranks))

        logging.info('Exact match: {}%'.format(100 * (1 - exact_match_errors / len(test_data))))
        logging.info('Structural exact match: {}%'.format(100 * (1 - structural_match_errors / len(test_data))))
        logging.info('Average gold log likelihood: {}'.format(sum(gold_likelihood_values)/len(test_data)))
        logging.info('Average gold probability: {}'.format(sum(gold_probs)/len(test_data)))
        logging.info('Gold output sequence is a candidate: {}%'.format(
            100 * float(len(gold_ranks))/len(test_data)))
        logging.info('Average rank of gold output sequence (when present): {}'.format(avg_gold_rank))
        logging.info('Input sequence is a candidate: {}%'.format(
            100 * float(len(input_copy_ranks))/len(test_data)))
        logging.info('Average rank of input sequence (when present): {}'.format(avg_input_copy_rank))
        logging.info('Original avg sentence bleu: {}'.format(orig_sentence_bleu))
        logging.info('Prediction avg sentence bleu: {}'.format(pred_sentence_bleu))

    for stat, val in edit_evaluator.evaluation_statistics().items():
        logging.info('{}: {}'.format(stat, val))

    logging.info('Total: {}'.format(len(test_data)))
    logging.info('Model: {}'.format(model_name))

def compute_bleu(references, hypotheses):
    sentence_scores = []
    for ref, hyp in zip(references, hypotheses):
        sentence_scores.append(sentence_bleu(ref, hyp, smoothing_function=SMOOTHING_FUNCTION))
    return 100*sum(sentence_scores)/len(sentence_scores)

def get_token_types(tokens):
    token_types = []
    for token in tokens:
        if re.match('[a-zA-Z_][a-zA-Z0-9_]*', token) or token == '%UNK%':
            token_types.append(IDENTIFER)
        elif is_num_literal(token):
            token_types.append(NUMBER)
        elif re.match(r'(["\'])(?:(?=(\\?))\2.)*?\1', token):
            token_types.append(STRING)
        else:
            token_types.append(token)
    return token_types

def compute_prediction_accuracy(predictions, test_data):
    num_errors = 0
    for i, (datasample, predictions) in enumerate(zip(test_data, predictions)):
        for used_prediction_idx in range(len(predictions)):
            # Since we are testing on edits, predicting the input_sequence is always wrong. Thus skip it!
            if predictions[used_prediction_idx][0] != datasample.input_sequence:
                break
        else:
            raise Exception('All output sequences are identical to input sequences. This cannot happen.')

        if predictions[used_prediction_idx][0] != datasample.output_sequence:
            num_errors += 1
    print(f'Matched {100 * (1 - num_errors / len(test_data))}% samples.')

def is_num_literal(token: str) -> bool:
    try:
        # Numeric literals come in too many flavors, use Python's tokenizer
        return next(tokenize.generate_tokens(StringIO(token).readline)).type == tokenize.NUMBER
    except:
        return False

def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)

    model_path = RichPath.create(arguments['MODEL_FILENAME'], azure_info_path)

    if arguments['--cpu']:
        model = BaseComponent.restore_model(model_path, 'cpu')
    else:
        model = BaseComponent.restore_model(model_path)

    test_data_path = RichPath.create(arguments['TEST_DATA'], azure_info_path)
    test_data = load_data_by_type(test_data_path, arguments['--data-type'])
    test_size = arguments.get('--test-size')

    if not test_size:
        test_size = len(test_data)
    else:
        test_size = int(test_size)
    test_data = test_data[:test_size]

    logging.info('Running test on %d examples', test_size)

    test(model, test_data, model.name, arguments['--data-type'] , not arguments['--no-prediction'],
    arguments['--greedy'], arguments['--verbose'])

if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
