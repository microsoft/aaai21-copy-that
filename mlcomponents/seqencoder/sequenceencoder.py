from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List, Union

import torch

from dpu_utils.ptutils import BaseComponent
from mlcomponents.embeddings import SequenceEmbedder


class SequenceEncoder(BaseComponent, ABC):
    """
    A general encoder of sequences.
    """

    def __init__(self, name: str, token_embedder: SequenceEmbedder,
                 hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        super(SequenceEncoder, self).__init__(name, hyperparameters)
        self.__token_embedder = token_embedder  # type: SequenceEmbedder

    @property
    @abstractmethod
    def summary_state_size(self) -> int:
        pass

    @property
    @abstractmethod
    def output_states_size(self) -> int:
        pass

    @property
    def token_embedder(self) -> SequenceEmbedder:
        return self.__token_embedder

    def _load_metadata_from_sample(self, data_to_load: List[str]) -> None:
        self.token_embedder.load_metadata_from_sample(data_to_load)

    def load_data_from_sample(self, data_to_load: List[str]) -> Optional[Any]:
        return self.token_embedder.load_data_from_sample(data_to_load)

    def initialize_minibatch(self) -> Dict[str, Any]:
        return self.token_embedder.initialize_minibatch()

    def extend_minibatch_by_sample(self, datapoint: Any, accumulated_minibatch_data: Dict[str, Any]) -> bool:
        return self.token_embedder.extend_minibatch_by_sample(datapoint, accumulated_minibatch_data)

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.token_embedder.finalize_minibatch(accumulated_minibatch_data)

    @abstractmethod
    def forward(self, *, input_sequence_data: Dict[str, Any], return_embedded_sequence: bool=False)\
        -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        :param input_sequence_data:
        :return: outputs: B x 2 * hidden_size
                 lengths: B
                 hn: batch x summary_output_dim or batch x 2 * num_layers * hidden_size
        """
        pass

    def get_summary(self, *, input_sequence_data: Dict[str, Any]) -> torch.Tensor:
        """
        Returns a BxD output of summaries.
        """
        with torch.no_grad():
            return self.forward(input_sequence_data=input_sequence_data)[2].cpu().numpy()
