import logging
import numpy as np

from dpu_utils.utils import run_and_debug, RichPath

from data.representationviz import RepresentationsVisualizer
from data.synthetic.charedits import get_dataset
from editrepcomponents.alignededitencoder import AlignedEditTokensEmbedding
from dpu_utils.ptutils import ComponentTrainer, BaseComponent
from mlcomponents.seqencoder import BiGruSequenceEncoder
from editrepcomponents.copyeditor import CopyEditor
from mlcomponents.embeddings import TokenSequenceEmbedder
from mlcomponents.seqdecoding import GruCopyingDecoder
from mlcomponents.seqdecoding import LuongAttention

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')


def run():
    dataset = get_dataset()
    logging.info('Generated %s synthetic datapoints.', len(dataset))

    training_set, validation_set = dataset[:int(.8 * len(dataset))], dataset[int(.8 * len(dataset)):]

    seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder', hyperparameters={'max_seq_length': 12, 'dropout_rate':0})
    input_sequence_encoder = BiGruSequenceEncoder('BiGruEncoder',
                                                  token_embedder=seq_embeddings,
                                                  hyperparameters={
                                                'num_layers':2,
                                                'hidden_size': 61,
                                            })

    attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})

    edit_token_embeddings = AlignedEditTokensEmbedding('EditEncoder', token_encoder=seq_embeddings)
    edit_encoder = BiGruSequenceEncoder('BiGruEditEncoder',
                                        token_embedder=edit_token_embeddings,
                                        hyperparameters={
                                     'num_layers':2,
                                 })
    decoder = GruCopyingDecoder('GruCopyingDecoder',
                       token_encoder=seq_embeddings,
                       standard_attention=attention,
                       hyperparameters={'initial_state_size': 244+128,
                                        'memories_hidden_dimension': 122,
                                        'dropout_rate':0,
                                        'additional_inputs_size':64*2,
                                        'max_memories_length': 12})

    model = CopyEditor('CopyEditor',
                       input_sequence_encoder=input_sequence_encoder,
                       edit_encoder=edit_encoder,
                       output_sequence_decoder=decoder,
                       learn_bidirectional_edits=True
                       )

    save_path = RichPath.create('./testmodel-copyseq2seq.pkl.gz')
    trainer = ComponentTrainer(model, save_path, max_num_epochs=100, minibatch_size=500)
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
            predictions.extend([s[0] for s in model.beam_decode(input_sequences=mb_data['input_sequences'], aligned_edits=mb_data['aligned_edits'],
                                                   ground_input_sequences=ground_input_sequences[start_idx:start_idx+num_elements])])
            start_idx += num_elements
            representations.extend(model.edit_encoder.get_summary(input_sequence_data=mb_data['aligned_edits']))
        if not is_full:
            break

    assert len(all_data) == len(predictions)

    num_errors = 0
    for i, (datasample, predictions) in enumerate(zip(validation_set, predictions)):
        if predictions[0] != datasample.output_sequence:
            print(datasample, predictions)
            num_errors += 1
    print(f'Matched {100 * (1 - num_errors/len(validation_set))}% samples.')

    representations = np.array(representations)
    viz = RepresentationsVisualizer(labeler=lambda d:d.edit_type[0])
    viz.print_nearest_neighbors(validation_set, representations, num_items=20)
    viz.plot_tsne(validation_set, representations, save_file='out.pdf')

if __name__ == '__main__':
    run_and_debug(run, True)
