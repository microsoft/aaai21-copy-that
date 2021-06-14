import logging
from typing import Optional, Dict, Any, List, Tuple, NamedTuple

import torch

from data.edits import Edit
from dpu_utils.ptutils import BaseComponent
from mlcomponents.seqdecoding import SeqDecoder
from mlcomponents.seqencoder import SequenceEncoder


class EncoderDecoder(BaseComponent):
    LOGGER = logging.getLogger('EncoderDecoder')

    def __init__(self, name: str, input_sequence_encoder: SequenceEncoder,
                 output_sequence_decoder: SeqDecoder,
                 hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        super(EncoderDecoder, self).__init__(name, hyperparameters)
        self.__input_sequence_encoder = input_sequence_encoder
        self.__output_sequence_decoder = output_sequence_decoder

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return { }

    def _finalize_component_metadata_and_model(self) -> None:
        pass

    @property
    def input_sequence_encoder(self):
        return self.__input_sequence_encoder

    @property
    def output_sequence_decoder(self):
        return self.__output_sequence_decoder


    def _load_metadata_from_sample(self, data_to_load: Edit) -> None:
        self.__input_sequence_encoder.load_metadata_from_sample(data_to_load.input_sequence)
        self.__output_sequence_decoder.load_metadata_from_sample(SeqDecoder.InputOutputSequence(
            input_sequence=data_to_load.input_sequence,
            output_sequence=data_to_load.output_sequence
        ))

    TensorizedData = NamedTuple('EncoderDecoderTensorizedData', [
        ('input_sequence', Any),
        ('output_sequence', Any),
    ])

    def load_data_from_sample(self, data_to_load: Edit) -> Optional['EncoderDecoder.TensorizedData']:
        return self.TensorizedData(
            input_sequence=self.__input_sequence_encoder.load_data_from_sample([SeqDecoder.START] + data_to_load.input_sequence +  [SeqDecoder.END]),
            output_sequence=self.__output_sequence_decoder.load_data_from_sample(SeqDecoder.InputOutputSequence(
                                                                                    input_sequence=[SeqDecoder.START] + data_to_load.input_sequence +  [SeqDecoder.END],
                                                                                    output_sequence=data_to_load.output_sequence
                                                                                ))
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            'input_sequences': self.__input_sequence_encoder.initialize_minibatch(),
            'output_sequences': self.__output_sequence_decoder.initialize_minibatch(),
        }

    def extend_minibatch_by_sample(self, datapoint: 'EncoderDecoder.TensorizedData', accumulated_minibatch_data: Dict[str, Any]) -> bool:
        continue_extending = self.__input_sequence_encoder.extend_minibatch_by_sample(
            datapoint=datapoint.input_sequence,
            accumulated_minibatch_data=accumulated_minibatch_data['input_sequences'])
        continue_extending &= self.__output_sequence_decoder.extend_minibatch_by_sample(
            datapoint=datapoint.output_sequence,
            accumulated_minibatch_data=accumulated_minibatch_data['output_sequences'])

        return continue_extending

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'input_sequences': self.__input_sequence_encoder.finalize_minibatch(accumulated_minibatch_data['input_sequences']),
            'output_sequences': self.__output_sequence_decoder.finalize_minibatch(accumulated_minibatch_data['output_sequences'])
        }

    def forward(self, *, input_sequences: Dict[str, Any], output_sequences: Dict[str, Any]):
        input_encoding = self.__input_sequence_encoder.forward(
                input_sequence_data=input_sequences,
                return_embedded_sequence=True
            )

        memories, memories_lengths, output_state, input_sequence_token_embeddings = input_encoding
        decoder_loss = self.__output_sequence_decoder.forward(memories=memories, memories_lengths=memories_lengths,
                                                              initial_state=output_state,
                                                              input_sequence_token_embeddings=input_sequence_token_embeddings,
                                                              **output_sequences)
        return decoder_loss

    def greedy_decode(self, input_sequences: Dict[str, Any],
                      ground_input_sequences: List[List[str]], max_length: int=50) -> List[Tuple[List[List[str]], List[float]]]:
        with torch.no_grad():
            ground_input_sequences, initial_state, memories, memory_lengths = self.__prepare_decoding(ground_input_sequences,
                                                                                                      input_sequences)

            return self.__output_sequence_decoder.greedy_decode(memories, memory_lengths,
                                                                initial_state=initial_state, max_length=max_length,
                                                                memories_str_representations=[[SeqDecoder.START] + g +  [SeqDecoder.END] for g in ground_input_sequences])

    def beam_decode(self, input_sequences: Dict[str, Any],
                    ground_input_sequences: List[List[str]], max_length: int=150) -> List[Tuple[List[List[str]], List[float]]]:
        with torch.no_grad():
            ground_input_sequences, initial_state, memories, memory_lengths = self.__prepare_decoding(ground_input_sequences,
                                                                                                      input_sequences)

            return self.__output_sequence_decoder.beam_decode(memories, memory_lengths,
                                                              initial_state=initial_state, max_length=max_length,
                                                              memories_str_representations=[[SeqDecoder.START] + g +  [SeqDecoder.END] for g in ground_input_sequences],
                                                              )


    def __prepare_decoding(self, ground_input_sequences, input_sequences):
        memories, memory_lengths, output_state = self.__input_sequence_encoder.forward(
            input_sequence_data=input_sequences)
        return ground_input_sequences, output_state, memories, memory_lengths

    def compute_likelihood(self, *, input_sequences: Dict[str, Any], output_sequences: Dict[str, Any],
                           return_debug_info: bool = False):
        with torch.no_grad():
            memories, memories_lengths, output_state = self.__input_sequence_encoder.forward(input_sequence_data=input_sequences)
            return self.__output_sequence_decoder.compute_likelihood(memories=memories,
                                                                memories_lengths=memories_lengths,
                                                                initial_state=output_state,
                                                                return_debug_info= return_debug_info,
                                                                **output_sequences)
