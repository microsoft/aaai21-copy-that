from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, NamedTuple

from dpu_utils.ptutils import BaseComponent
from mlcomponents.embeddings import SequenceEmbedder


class SeqDecoder(BaseComponent, ABC):
    def __init__(self, name: str, token_encoder: SequenceEmbedder,
                 hyperparameters: Optional[Dict[str, Any]] = None
                 ) -> None:
        super(SeqDecoder, self).__init__(name, hyperparameters)
        self.__target_token_encoder = token_encoder

    START = '[CLS]'
    END = '[SEP]'

    InputOutputSequence = NamedTuple('InputOutputSequence', [
        ('input_sequence', List[str]),
        ('output_sequence', List[str]),
    ])

    @property
    def target_token_encoder(self):
        return self.__target_token_encoder

    def _add_start_end(self, sequence: List[str]) -> List[str]:
        return [SeqDecoder.START] + sequence + [SeqDecoder.END]

    def _load_metadata_from_sample(self, data_to_load: 'SeqDecoder.InputOutputSequence') -> None:
        self.target_token_encoder.load_metadata_from_sample(self._add_start_end(data_to_load.output_sequence))

    @abstractmethod
    def load_data_from_sample(self, data_to_load: 'SeqDecoder.InputOutputSequence') -> Any:
        pass

    def _reset_component_metrics(self) -> None:
        self._num_minibatches = 0
        self._loss_sum = 0

    def _component_metrics(self) -> Dict[str, float]:
        return {
            'Total Decoder Loss': self._loss_sum / self._num_minibatches
        }

    @abstractmethod
    def compute_likelihood(self, *, memories, memories_lengths, initial_state, additional_decoder_input,
                           return_debug_info: bool = False, **kwargs):
        pass

    @abstractmethod
    def greedy_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                      max_length: int = 40, additional_decoder_input = None) -> List[Tuple[List[List[str]], List[float]]]:
        pass

    @abstractmethod
    def beam_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                    max_length: int = 40, max_beam_size: int = 5, additional_decoder_input = None) -> List[Tuple[List[List[str]], List[float]]]:
        pass
