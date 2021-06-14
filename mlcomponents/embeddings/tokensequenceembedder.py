import logging
from collections import Counter
import typing
from typing import Optional, Dict, Any, List, NamedTuple

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
from torch import nn

from mlcomponents.embeddings.sequenceembedder import SequenceEmbedder


class TokenSequenceEmbedder(SequenceEmbedder):
    """
    Component that converts a list of tokens into a fixed-size matrix of embeddings.
    """

    LOGGER = logging.getLogger('TokenSequenceEmbedder')

    def __init__(self, name: str, hyperparameters: Optional[Dict[str, Any]]=None) -> None:
        super(TokenSequenceEmbedder, self).__init__(name, hyperparameters)
        self.__metadata_token_counter = None  # type: Optional[typing.Counter[str]]
        self.__vocabulary = None  # type: Optional[Vocabulary]
        self.__embedding_layer = None # type: Optional[nn.Embedding]
        self.__dropout_layer = None  # type: Optional[nn.Dropout]

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {'embedding_size': 64,
                'max_vocabulary_size': 10000,
                'min_word_count_threshold': 7,
                'max_seq_length': 30,
                'dropout_rate': 0.2,
                }

    @property
    def embedding_size(self) -> int:
        return self.get_hyperparameter('embedding_size')

    @property
    def embedding_matrix(self) -> torch.Tensor:
        assert self.__embedding_layer is not None, 'Embeddings have not been initialized.'
        return self.__embedding_layer.weight

    @property
    def vocabulary(self) -> Vocabulary:
        return self.__vocabulary

    # region Metadata Loading
    def _init_component_metadata(self) -> None:
        if self.__metadata_token_counter is None:
            self.__metadata_token_counter = Counter()

    def _load_metadata_from_sample(self, data_to_load: List[str]) -> None:
        self.__metadata_token_counter.update(data_to_load[:self.get_hyperparameter('max_seq_length')])

    def _finalize_component_metadata_and_model(self) -> None:
        if self.__metadata_token_counter is None or self.__vocabulary is not None:
            return # This module has already been finalized
        token_counter = self.__metadata_token_counter
        self.__metadata_token_counter = None

        self.__vocabulary = Vocabulary.create_vocabulary(tokens=token_counter,
                                                         max_size=self.get_hyperparameter('max_vocabulary_size'),
                                                         count_threshold=self.get_hyperparameter('min_word_count_threshold'),
                                                         add_pad=True)
        self.LOGGER.info('Vocabulary Size of %s is %s', self.name, len(self.__vocabulary))
        self.__embedding_layer = nn.Embedding(num_embeddings=len(self.__vocabulary),
                                              embedding_dim=self.get_hyperparameter('embedding_size'),
                                              padding_idx=self.__vocabulary.get_id_or_unk(Vocabulary.get_pad()))
        self.__dropout_layer = nn.Dropout(p=self.get_hyperparameter('dropout_rate'))

    # endregion

    TensorizedData = NamedTuple('EmbeddingTensorizedData', [
        ('token_ids', np.ndarray),
        ('length', int)
    ])

    def load_data_from_sample(self, data_to_load: List[str]) -> Optional['TokenSequenceEmbedder.TensorizedData']:
        return self.TensorizedData(
            token_ids=np.array(self.convert_sequence_to_tensor(data_to_load), dtype=np.int32),
            length=min(len(data_to_load), self.get_hyperparameter('max_seq_length'))
        )

    def convert_sequence_to_tensor(self, token_sequence: List[str]):
        return self.__vocabulary.get_id_or_unk_multiple(
            tokens=[Vocabulary.get_pad() if t is None else t for t in token_sequence[:self.get_hyperparameter('max_seq_length')]]
        )

    # region Minibatching
    def initialize_minibatch(self) -> Dict[str, Any]:
        return {'token_sequence_ids': [], 'lengths': []}

    def extend_minibatch_by_sample(self, datapoint: 'TokenSequenceEmbedder.TensorizedData', accumulated_minibatch_data: Dict[str, Any]) -> bool:
        accumulated_minibatch_data['token_sequence_ids'].append(datapoint.token_ids)
        accumulated_minibatch_data['lengths'].append(datapoint.length)
        return True

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        accumulated_token_ids = accumulated_minibatch_data['token_sequence_ids']
        max_size = np.max(accumulated_minibatch_data['lengths'])

        token_ids = np.zeros((len(accumulated_token_ids), max_size), dtype=np.int32)
        for i in range(len(accumulated_token_ids)):
            token_ids[i, :len(accumulated_token_ids[i])] = accumulated_token_ids[i]
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.int64, device=self.device),
            'lengths': torch.tensor(accumulated_minibatch_data['lengths'], dtype=torch.int64, device=self.device)
        }
    # endregion

    def _compute_embeddings(self, token_ids: torch.Tensor, lengths: torch.Tensor, add_sequence_related_annotations: bool):
        embedded = self.__embedding_layer(token_ids)  # B x max_len x D
        return self.__dropout_layer(embedded)
