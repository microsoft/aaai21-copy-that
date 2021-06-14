import logging
from typing import Set

from dpu_utils.utils import run_and_debug, RichPath

from data.edits import Edit
from dpu_utils.ptutils import ComponentTrainer
from mlcomponents.seqencoder import BiGruSequenceEncoder
from mlcomponents.embeddings import TokenSequenceEmbedder
from mlcomponents.encoderdecoder import EncoderDecoder
from mlcomponents.seqdecoding import GruDecoder
from mlcomponents.seqdecoding import LuongAttention

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')


all_letters = [chr(65+i) for i in range(26)] + [chr(97+i) for i in range(26)] \
              + [chr(913+i) for i in range(25)] +[chr(945+i) for i in range(25)]  # Add greek letters

def get_at(pos: int, length: int) -> Edit:
    before = tuple(all_letters[j % len(all_letters)] for j in range(pos, pos + length))
    after = tuple(all_letters[j % len(all_letters)] for j in range(pos + length, pos + length * 2))
    return Edit(input_sequence=before, output_sequence=after, edit_type='', provenance='')

def get_dataset() -> Set[Edit]:
    dataset = set()  # type: Set[Edit]
    for i in range(len(all_letters)):
        for l in range(2, 6):
            dataset.add(get_at(i, l))
    return dataset


def run():
    dataset = list(Edit(
        input_sequence=list(e.input_sequence),
        output_sequence=list(e.output_sequence),
        edit_type=e.edit_type,
        provenance=e.provenance
    ) for e in get_dataset())

    seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder', hyperparameters={'max_seq_length': 7, 'dropout_rate': 0})
    input_encoder = BiGruSequenceEncoder('BiGruEncoder', seq_embeddings)
    attention = LuongAttention('StandardAttention', hyperparameters={
        'memories_hidden_dimension': input_encoder.output_states_size
    })
    decoder = GruDecoder('GruDecoder', seq_embeddings, standard_attention=attention,
                       hyperparameters={
                           'initial_state_size': 64
                       })

    model = EncoderDecoder('EncoderDecoder',
        input_sequence_encoder=input_encoder,
        output_sequence_decoder=decoder
    )

    trainer = ComponentTrainer(model, RichPath.create('./testmodel.pkl.gz'), max_num_epochs=500)
    trainer.train(dataset, dataset, patience=50)

    ## Try greedy decoding
    model = trainer.model
    model.eval()
    all_data = [model.load_data_from_sample(d) for d in dataset]
    data_iter = iter(all_data)
    predictions = []
    is_full = True
    while is_full:
        mb_data, is_full, _ = model.create_minibatch(data_iter, max_num_items=100)
        predictions.extend(model.greedy_decode(input_sequences=mb_data['input_sequences']))
        if not is_full:
            break

    assert len(all_data) == len(predictions)

    num_errors = 0
    for i, (datasample, predictions) in enumerate(zip(dataset, predictions)):
        if predictions[0] != datasample.output_sequence:
            print(f'{i} not matching data sample: {datasample}')
            print(f'Predicted: {predictions}')
            print('----------')
            num_errors += 1
    print(f'Matched {100 * (1 - num_errors/len(dataset))}% samples.')

run_and_debug(run, True)
