from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Any

import torch
from dpu_utils.mlutils import Vocabulary
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from dpu_utils.ptutils import BaseComponent


class SequenceEmbedder(BaseComponent, ABC):
    @property
    @abstractmethod
    def embedding_size(self) -> int:
        pass

    @property
    @abstractmethod
    def vocabulary(self) -> Vocabulary:
        pass

    @property
    @abstractmethod
    def embedding_matrix(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_embeddings(self, token_ids: torch.Tensor, lengths: torch.Tensor, add_sequence_related_annotations: bool):
        pass

    @abstractmethod
    def _load_metadata_from_sample(self, data_to_load: List[str]) -> None:
        pass

    @abstractmethod
    def load_data_from_sample(self, data_to_load: List[str]) -> Any:
        pass

    def forward(self, *, token_ids: torch.Tensor, lengths: torch.Tensor, as_packed_sequence: bool=True,
                add_sequence_related_annotations: bool=False) -> Union[Tuple[PackedSequence, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert a B x max_len matrix of integer ids to B x max_len x embedding_size
        :param input_token_ids: ? x 'embedding_size'
        :param add_sequence_related_annotations Add any sequence_related_annotations (e.g. in positional encodings)
        :return: a PackedSequence with batch_first=True and the indices to scatter things back into their original order
            if as_packed_sequence=True, otherwise, a tuple of the (embedded tokens, vector of input lengths).
        """
        embedded = self._compute_embeddings(token_ids, lengths, add_sequence_related_annotations)  # B x max_len x D
        if not as_packed_sequence:
            return embedded, lengths
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        # The reverse map, to restore the original order (over batches)
        reverse_map = torch.zeros_like(indices, device=self.device)\
            .scatter_(dim=0, index=indices, src=torch.arange(indices.shape[0], device=self.device))  # B

        return pack_padded_sequence(embedded[indices], sorted_lengths, batch_first=True), reverse_map
