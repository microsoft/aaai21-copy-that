from typing import Optional

from pytorch_transformers import BertConfig

from editrepcomponents.alignededitencoder import AlignedEditTokensEmbedding
from mlcomponents.seqdecoding.spancopydecoder import GruSpanCopyingDecoder
from mlcomponents.seqencoder import BiGruSequenceEncoder
from editrepcomponents.copyeditor import CopyEditor
from mlcomponents.embeddings import TokenSequenceEmbedder
from mlcomponents.seqdecoding import GruCopyingDecoder, GruDecoder
from mlcomponents.seqdecoding import LuongAttention
from mlcomponents.encoderdecoder import EncoderDecoder


def create_copy_seq2seq_model(bidirectional: bool=False) -> CopyEditor:
    """A Seq2Seq Editor Model with Attention and Copying"""
    seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder',
                                             hyperparameters={'max_seq_length': 50, 'min_word_count_threshold': 11, })
    input_sequence_encoder = BiGruSequenceEncoder('BiGruInputEncoder',
                                                  token_embedder=seq_embeddings,
                                                  hyperparameters={
                                               'num_layers': 2,
                                               'hidden_size': 64,
                                           })

    attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})

    edit_token_embeddings = AlignedEditTokensEmbedding('EditEncoder', token_encoder=seq_embeddings)
    edit_encoder = BiGruSequenceEncoder('BiGruEditEncoder',
                                        token_embedder=edit_token_embeddings,
                                        hyperparameters={
                                     'num_layers': 2,
                                     'hidden_size': 64,
                                 })
    decoder = GruCopyingDecoder('GruCopyDecoder',
                                token_encoder=seq_embeddings,
                                standard_attention=attention,
                                hyperparameters={'initial_state_size':
                                                    edit_encoder.get_hyperparameter('hidden_size') *
                                                    edit_encoder.get_hyperparameter('num_layers') * 2 +
                                                    input_sequence_encoder.get_hyperparameter('hidden_size') *
                                                    input_sequence_encoder.get_hyperparameter('num_layers') * 2,
                                                 'memories_hidden_dimension': 2 * input_sequence_encoder.get_hyperparameter('hidden_size'),
                                                 'additional_inputs_size':
                                                    edit_encoder.get_hyperparameter('hidden_size') *
                                                    edit_encoder.get_hyperparameter('num_layers') * 2,
                                                 'max_memories_length': seq_embeddings.get_hyperparameter('max_seq_length')})

    model = CopyEditor('Editor',
                       input_sequence_encoder=input_sequence_encoder,
                       edit_encoder=edit_encoder,
                       output_sequence_decoder=decoder,
                       learn_bidirectional_edits=bidirectional
                       )
    return model

def create_base_copy_seq2seq_model(pre_trained_seq_embeddings = None, pre_trained_gru = None) -> EncoderDecoder:
    """A Seq2Seq Editor Model with Attention and Copying"""
    if pre_trained_seq_embeddings is None:
        seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder',
                                                hyperparameters={'max_seq_length': 50, 'min_word_count_threshold': 11})
    else:
        seq_embeddings = pre_trained_seq_embeddings
    input_sequence_encoder = BiGruSequenceEncoder('BiGruInputEncoder',
                                                  token_embedder=seq_embeddings,
                                                  hyperparameters={
                                               'num_layers': 2,
                                               'hidden_size': 128,
                                           })

    attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})

    decoder = GruCopyingDecoder('GruCopyDecoder',
                                token_encoder=seq_embeddings,
                                standard_attention=attention,
                                hyperparameters={'initial_state_size':
                                                    input_sequence_encoder.get_hyperparameter('hidden_size') *
                                                    input_sequence_encoder.get_hyperparameter('num_layers') * 2,
                                                 'memories_hidden_dimension': 2 * input_sequence_encoder.get_hyperparameter('hidden_size'),
                                                 'additional_inputs_size': 0,
                                                 'max_memories_length': seq_embeddings.get_hyperparameter('max_seq_length')})

    model = EncoderDecoder('Seq2SeqModel',
                       input_sequence_encoder=input_sequence_encoder,
                       output_sequence_decoder=decoder
                       )
    return model


def create_seq2seq_with_span_copy_model(bidirectional: bool=False):
    """A Seq2Seq Editor Model with Attention and Copying"""
    seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder',
                                           hyperparameters={
                                               'max_seq_length': 50,
                                               'min_word_count_threshold': 11,
                                               'max_vocabulary_size': 25000
                                            })
    input_sequence_encoder = BiGruSequenceEncoder('BiGruInputEncoder',
                                                  token_embedder=seq_embeddings,
                                                  hyperparameters={
                                                      'num_layers': 2,
                                                      'hidden_size': 64,
                                                  })

    attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})
    copy_attention = LuongAttention('StandardAttention',
                               hyperparameters={'memories_hidden_dimension': input_sequence_encoder.output_states_size})

    edit_token_embeddings = AlignedEditTokensEmbedding('EditEncoder', token_encoder=seq_embeddings)
    edit_encoder = BiGruSequenceEncoder('BiGruEditEncoder',
                                        token_embedder=edit_token_embeddings,
                                        hyperparameters={
                                            'num_layers': 2,
                                            'hidden_size': 64,
                                        })
    decoder = GruSpanCopyingDecoder('GruCopyDecoder',
                                token_encoder=seq_embeddings,
                                standard_attention=attention,
                                copy_attention=copy_attention,
                                hyperparameters={'initial_state_size':
                                                    edit_encoder.get_hyperparameter('hidden_size') *
                                                    edit_encoder.get_hyperparameter('num_layers') * 2 +
                                                    input_sequence_encoder.get_hyperparameter('hidden_size') *
                                                    input_sequence_encoder.get_hyperparameter('num_layers') * 2 ,
                                                 'memories_hidden_dimension': 2 * input_sequence_encoder.get_hyperparameter('hidden_size'),
                                                 'additional_inputs_size':
                                                    edit_encoder.get_hyperparameter('hidden_size') *
                                                    edit_encoder.get_hyperparameter('num_layers') * 2,
                                                 'max_memories_length': seq_embeddings.get_hyperparameter('max_seq_length')})


    model = CopyEditor('Editor',
                       input_sequence_encoder=input_sequence_encoder,
                       edit_encoder=edit_encoder,
                       output_sequence_decoder=decoder,
                       learn_bidirectional_edits=bidirectional
                       )
    return model

def create_gru_lm():
    seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder',
                                           hyperparameters={
                                               'max_seq_length': 50,
                                               'min_word_count_threshold': 11,
                                               'max_vocabulary_size': 30000,
                                               'embedding_size':256
                                           })
    decoder = GruDecoder('GruDecoder', seq_embeddings,
                         hyperparameters= {
                             'hidden_size': 128,
                             'initial_state_size': 128
                         },
                         include_summarizing_network=False)
    return decoder


def create_base_seq2seq_with_span_copy_model(pre_trained_seq_embeddings = None, pre_trained_gru = None):
    if pre_trained_seq_embeddings is None:
        seq_embeddings = TokenSequenceEmbedder('SeqTokenEmbedder',
                                            hyperparameters={
                                                'max_seq_length': 50,
                                                'min_word_count_threshold': 11,
                                                'max_vocabulary_size': 10000,
                                                'embedding_size': 128
                                                })
    else:
        seq_embeddings = pre_trained_seq_embeddings

    input_sequence_encoder = BiGruSequenceEncoder('BiGruInputEncoder',
                                                  token_embedder=seq_embeddings,
                                                  hyperparameters={
                                                      'num_layers': 1,
                                                      'hidden_size': 128,
                                                  })

    attention = LuongAttention('StandardAttention',
                               hyperparameters={
                                   'memories_hidden_dimension': input_sequence_encoder.output_states_size,
                                   'lookup_hidden_dimension': 128,
                                   'output_size': 128
                                   })
    copy_attention = LuongAttention('StandardAttention',
                               hyperparameters={
                                   'memories_hidden_dimension': input_sequence_encoder.output_states_size,
                                   'lookup_hidden_dimension': 128,
                                   'output_size': 128
                                })


    decoder = GruSpanCopyingDecoder('GruCopyDecoder',
                                token_encoder=seq_embeddings,
                                standard_attention=attention,
                                copy_attention=copy_attention,
                                pre_trained_gru=pre_trained_gru,
                                hyperparameters={'initial_state_size':
                                                    input_sequence_encoder.get_hyperparameter('hidden_size') *
                                                    input_sequence_encoder.get_hyperparameter('num_layers') * 2 ,
                                                 'memories_hidden_dimension': 2 * input_sequence_encoder.get_hyperparameter('hidden_size'),
                                                 'additional_inputs_size': 0,
                                                 'hidden_size': 128,
                                                 'max_memories_length': seq_embeddings.get_hyperparameter('max_seq_length')})


    model = EncoderDecoder('CopySpanModel',
                       input_sequence_encoder=input_sequence_encoder,
                       output_sequence_decoder=decoder,
                       )
    return model
