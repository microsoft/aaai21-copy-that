import logging
import random

import numpy as np

from dpu_utils.utils import run_and_debug, RichPath

from data.representationviz import RepresentationsVisualizer
from data.synthetic.charedits import get_dataset
from editrepcomponents.alignededitencoder import AlignedEditTokensEmbedding
from dpu_utils.ptutils import BaseComponent, ComponentTrainer
from mlcomponents.seqdecoding.spancopydecoder import GruSpanCopyingDecoder
from mlcomponents.seqencoder import BiGruSequenceEncoder
from editrepcomponents.copyeditor import CopyEditor
from mlcomponents.embeddings import TokenSequenceEmbedder
from mlcomponents.seqdecoding import LuongAttention

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')


def run():
    greedy_decoding = False

    np.random.seed(1)
    random.seed(1)
    dataset = get_dataset()
    logging.info('Generated %s synthetic datapoints.', len(dataset))

    training_set, validation_set = dataset[:int(.8 * len(dataset))], dataset[int(.8 * len(dataset)):]

    seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder', hyperparameters={'max_seq_length': 12, 'dropout_rate':0, 'min_word_count_threshold': 1})
    input_sequence_encoder = BiGruSequenceEncoder('BiGruEncoder',
                                                  token_embedder=seq_embeddings,
                                                  hyperparameters={
                                                'num_layers':2,
                                                'hidden_size': 61,
                                            })

    attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})
    copy_attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})

    edit_token_embeddings = AlignedEditTokensEmbedding('EditEncoder', token_encoder=seq_embeddings)
    edit_encoder = BiGruSequenceEncoder('BiGruEditEncoder',
                                        token_embedder=edit_token_embeddings,
                                        hyperparameters={
                                     'num_layers':3,
                                 })
    decoder = GruSpanCopyingDecoder('GruSpanCopyingDecoder',
                       token_encoder=seq_embeddings,
                       standard_attention=attention,
                       copy_attention=copy_attention,
                       hyperparameters={'initial_state_size': 244+192,
                                        'memories_hidden_dimension': 122,
                                        'dropout_rate':0,
                                        'additional_inputs_size':64*3,
                                        'max_memories_length': 12})

    model = CopyEditor('CopyEditor',
                       input_sequence_encoder=input_sequence_encoder,
                       edit_encoder=edit_encoder,
                       output_sequence_decoder=decoder,
                       learn_bidirectional_edits=True
                       )

    save_path = RichPath.create('./testmodel-copyspan.pkl.gz')
    trainer = ComponentTrainer(model, save_path, max_num_epochs=50, minibatch_size=500)
    trainer.train(training_set, validation_set, patience=10)

    ## Try greedy decoding
    model = None
    model = BaseComponent.restore_model(save_path)  # type: CopyEditor
    model.eval()
    all_data = [model.load_data_from_sample(d) for d in validation_set]
    ground_input_sequences = [d.input_sequence for d in validation_set]
    data_iter = iter(all_data)
    predictions = []
    representations = []
    is_full = True
    start_idx = 0
    while is_full:
        mb_data, is_full, num_elements = model.create_minibatch(data_iter, max_num_items=100)
        if num_elements > 0:
            if greedy_decoding:
                mb_predictions = [s for s in model.greedy_decode(input_sequences=mb_data['input_sequences'],
                                                                 aligned_edits=mb_data['aligned_edits'],
                                                                 ground_input_sequences=ground_input_sequences[
                                                                                         start_idx:start_idx + num_elements])]
            else:
                mb_predictions = [s for s in model.beam_decode(input_sequences=mb_data['input_sequences'], aligned_edits=mb_data['aligned_edits'],
                                                       ground_input_sequences=ground_input_sequences[start_idx:start_idx+num_elements])]
            predictions.extend(mb_predictions)
            start_idx += num_elements
            representations.extend(model.edit_encoder.get_summary(input_sequence_data=mb_data['aligned_edits']))
        if not is_full:
            break

    assert len(all_data) == len(predictions)

    num_errors_at_1 = 0
    num_errors_at_5 = 0
    for i, (datasample, predictions) in enumerate(zip(validation_set, predictions)):
        if predictions[0][0] != datasample.output_sequence:
            print(datasample, predictions)
            num_errors_at_1 += 1
        if not any(predictions[i][0] == datasample.output_sequence for i in range(len(predictions))):
            num_errors_at_5 += 1

    print(f'Matched @1 {100 * (1 - num_errors_at_1/len(validation_set))}% samples.')
    print(f'Matched @5 {100 * (1 - num_errors_at_5/len(validation_set))}% samples.')

    representations = np.array(representations)
    viz = RepresentationsVisualizer(labeler=lambda d:d.edit_type)
    viz.print_nearest_neighbors(validation_set, representations, num_items=20)
    # viz.plot_tsne(validation_set, representations, save_file='out.pdf')


run_and_debug(run, True)
