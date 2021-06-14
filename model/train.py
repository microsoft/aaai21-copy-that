#!/usr/bin/env python
"""
Usage:
    train.py [options] TRAIN_DATA_PATH VALID_DATA_PATH MODEL_TYPE TARGET_MODEL_FILENAME
    train.py [options] --split-valid TRAIN_DATA_PATH MODEL_TYPE TARGET_MODEL_FILENAME

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --data-type=<type>         The type of data to be used. Possible options fce, code, wikiatomicedits. [default: fce]
    --max-num-epochs=<epochs>  The maximum number of epochs to run training for. [default: 100]
    --minibatch-size=<size>    The minibatch size. [default: 200]
    --validation-pct=<pct>     The percent of the data to keep as validation if a validation set is not explicitly given. [default: 0.1]
    --restore_path=<path>      The path to previous model file for starting from previous checkpoint.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

from data.loading import load_data_by_type
from dpu_utils.ptutils import BaseComponent, ComponentTrainer
from model.editrepresentationmodels import create_copy_seq2seq_model, \
    create_seq2seq_with_span_copy_model, \
    create_base_copy_seq2seq_model, \
    create_base_seq2seq_with_span_copy_model, \
    create_gru_lm

ALL_MODELS = {
    'copyseq2seq': create_copy_seq2seq_model,
    'basecopyseq2seq': create_base_copy_seq2seq_model,
    'copyseq2seq-bidi': lambda: create_copy_seq2seq_model(bidirectional=True),
    'gru-lm': create_gru_lm,

    'basecopyspan': create_base_seq2seq_with_span_copy_model,
    'copyspan': create_seq2seq_with_span_copy_model,
    'copyspan-bidi': lambda:create_seq2seq_with_span_copy_model(bidirectional=True),
}


def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)

    training_data_path = RichPath.create(arguments['TRAIN_DATA_PATH'], azure_info_path)
    training_data = load_data_by_type(training_data_path, arguments['--data-type'], as_list=arguments['--split-valid'])

    if not arguments['--split-valid']:
        validation_data_path = RichPath.create(arguments['VALID_DATA_PATH'], azure_info_path)
        validation_data = load_data_by_type(validation_data_path, arguments['--data-type'], as_list=False)
    else:
        logging.info('No validation set provided. One will be carved out from the training set.')
        logging.warning('Lazy loading does not work when using --split-valid')
        validation_pct = 1 - float(arguments['--validation-pct'])
        assert 0 < validation_pct < 1, 'Validation Split should be in (0,1)'
        training_data, validation_data = training_data[:int(validation_pct * len(training_data))], training_data[int(validation_pct * len(training_data)):]

    model_path = RichPath.create(arguments['TARGET_MODEL_FILENAME'], azure_info_path)
    model_name = arguments['MODEL_TYPE']

    initialize_metadata = True
    restore_path = arguments.get('--restore_path', None)
    if restore_path:
        model = BaseComponent.restore_model(RichPath.create(restore_path, azure_info_path))
        initialize_metadata = False
    elif model_name in ALL_MODELS:
        model = ALL_MODELS[model_name]()
    else:
        raise ValueError(f'Unrecognized model tyoe {model_name}. Available names: {ALL_MODELS.keys()}')

    trainer = ComponentTrainer(model, model_path, max_num_epochs=int(arguments['--max-num-epochs']),
        minibatch_size=int(arguments['--minibatch-size']))

    trainer.train(training_data, validation_data, show_progress_bar=not arguments['--quiet'],
        initialize_metadata=initialize_metadata)


if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
