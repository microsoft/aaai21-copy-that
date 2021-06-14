from collections import Hashable
from typing import Optional, Dict, Any, NamedTuple

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from data.edits import ChangeType, sequence_diff, AlignedDiffRepresentation
from dpu_utils.ptutils import BaseComponent
from mlcomponents.embeddings import SequenceEmbedder


class AlignedEditTokensEmbedding(BaseComponent):
    """
    Given two sequences of tokens, compute the diff that
    aligns them and embed them.
    """
    def __init__(self, name: str, token_encoder: SequenceEmbedder,
                 hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        super(AlignedEditTokensEmbedding, self).__init__(name, hyperparameters)
        self.__token_encoder = token_encoder
        self.__change_type_embedding_layer = None  # type: Optional[nn.Embedding]

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'change_type_embedding_size': 8,
            'output_representation_size': None
        }

    @property
    def change_type_embedding_size(self) -> int:
        return self.get_hyperparameter('change_type_embedding_size')

    @property
    def token_encoder(self) -> SequenceEmbedder:
        return self.__token_encoder

    @property
    def embedding_size(self) -> int:
        return self.__token_encoder.embedding_size * 2 + self.get_hyperparameter('change_type_embedding_size')

    @property
    def change_type_embedding_matrix(self) -> torch.Tensor:
        assert self.__change_type_embedding_layer is not None, 'Embeddings have not been initialized.'
        return self.__change_type_embedding_layer.weight

    def _load_metadata_from_sample(self, data_to_load) -> None:
        self.__token_encoder.load_metadata_from_sample(data_to_load.output_sequence)
        self.__token_encoder.load_metadata_from_sample(data_to_load.input_sequence)

    def _finalize_component_metadata_and_model(self) -> None:
        self.__change_type_embedding_layer = nn.Embedding(num_embeddings=len(ChangeType),
                                                          embedding_dim=self.get_hyperparameter('change_type_embedding_size'),
                                                          )
        if self.get_hyperparameter('output_representation_size') is not None:
            self.__output_layer = nn.Linear(in_features=2*self.__token_encoder.embedding_size + self.get_hyperparameter('change_type_embedding_size'),
                                            out_features=self.get_hyperparameter('output_representation_size'))

    TensorizedData = NamedTuple('AlignedEditEncoderTensorizedData', [
       ('before_token_ids', np.ndarray),
       ('after_token_ids', np.ndarray),
       ('change_ids', np.ndarray),
       ('length', int)
     ])

    def load_data_from_sample(self, data_to_load: Any) -> Optional['AlignedEditTokensEmbedding.TensorizedData']:
        max_seq_length = self.__token_encoder.get_hyperparameter('max_seq_length')

        pad_id = self.__token_encoder.load_data_from_sample([Vocabulary.get_pad()]).token_ids[0]

        aligned_edit_representation = sequence_diff(data_to_load.input_sequence, data_to_load.output_sequence)

        change_types=[change_type.value for change_type in aligned_edit_representation.change_type[:max_seq_length]]
        before_tokens=self.__token_encoder.load_data_from_sample(aligned_edit_representation.before_tokens).token_ids[:max_seq_length]
        after_tokens=self.__token_encoder.load_data_from_sample(aligned_edit_representation.after_tokens).token_ids[:max_seq_length]

        assert len(change_types) == len(before_tokens) == len(after_tokens)

        diff = self.TensorizedData(
            before_token_ids=np.array(before_tokens, dtype=np.int32),
            after_token_ids=np.array(after_tokens, dtype=np.int32),
            change_ids=np.array(change_types, dtype=np.int8),
            length=len(change_types)
        )
        return diff

    def initialize_minibatch(self):
        return {'before_token_ids': [],
                'after_token_ids': [],
                'change_ids': [],
                'lengths': []
                }

    def extend_minibatch_by_sample(self, datapoint: 'AlignedEditTokensEmbedding.TensorizedData', accumulated_minibatch_data: Dict[str, Any]) -> bool:
        accumulated_minibatch_data['before_token_ids'].append(datapoint.before_token_ids)
        accumulated_minibatch_data['after_token_ids'].append(datapoint.after_token_ids)
        accumulated_minibatch_data['change_ids'].append(datapoint.change_ids)
        accumulated_minibatch_data['lengths'].append(datapoint.length)
        return True

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        lengths = accumulated_minibatch_data['lengths']
        accumulated_token_ids_before = accumulated_minibatch_data['before_token_ids']
        accumulated_token_ids_after = accumulated_minibatch_data['after_token_ids']
        accumulated_change_ids = accumulated_minibatch_data['change_ids']

        max_seq_size = max(lengths)
        batch_size = len(lengths)

        token_ids_before = np.zeros((batch_size, max_seq_size), dtype=np.int32)
        token_ids_after = np.zeros((batch_size, max_seq_size), dtype=np.int32)
        change_ids = np.zeros((batch_size, max_seq_size), dtype=np.int32)
        for i in range(batch_size):
            example_length = lengths[i]
            token_ids_before[i, :example_length] = accumulated_token_ids_before[i]
            token_ids_after[i, :example_length] = accumulated_token_ids_after[i]
            change_ids[i, :example_length] = accumulated_change_ids[i]

        return {
            'token_ids_before': torch.tensor(token_ids_before, dtype=torch.int64, device=self.device),
            'token_ids_after':  torch.tensor(token_ids_after, dtype=torch.int64, device=self.device),
            'change_ids': torch.tensor(change_ids, dtype=torch.int64, device=self.device),
            'lengths': torch.tensor(lengths, dtype=torch.int64, device=self.device)
        }

    def forward(self, *, token_ids_before: torch.Tensor, token_ids_after: torch.Tensor, change_ids: torch.Tensor,
                lengths: torch.Tensor, as_packed_sequence: bool=True, add_sequence_related_annotations: bool=True):
        embedded_tokens_before, lengths = self.__token_encoder.forward(token_ids=token_ids_before,
                                                                       lengths=lengths, as_packed_sequence=False,
                                                                       add_sequence_related_annotations=add_sequence_related_annotations)
        embedded_tokens_after, _ = self.__token_encoder.forward(token_ids=token_ids_after, lengths=lengths,
                                                                as_packed_sequence=False,
                                                                add_sequence_related_annotations=add_sequence_related_annotations)

        change_embeddings = self.__change_type_embedding_layer(change_ids)  # B x D_c

        embeddings = torch.cat([embedded_tokens_before, embedded_tokens_after, change_embeddings], dim=-1)  # B x (2D + D_c)

        if self.get_hyperparameter('output_representation_size') is not None:
            embeddings = self.__output_layer(embeddings)

        if not as_packed_sequence:
            return embeddings, lengths

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        # The reverse map, to restore the original order (over batches)
        reverse_map = torch.zeros_like(indices).scatter_(dim=0, index=indices, src=torch.arange(indices.shape[0], device=self.device))  # B
        return pack_padded_sequence(embeddings[indices], sorted_lengths, batch_first=True), reverse_map
