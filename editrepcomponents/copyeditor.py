import logging
from typing import Optional, Dict, Any, List, Tuple, NamedTuple

import torch
from torch import nn

from data.edits import Edit
from dpu_utils.ptutils import BaseComponent
from mlcomponents.seqdecoding import SeqDecoder
from mlcomponents.seqencoder import SequenceEncoder


class CopyEditor(BaseComponent):

    LOGGER = logging.getLogger('CopyEditor')

    def __init__(self, name: str, input_sequence_encoder: SequenceEncoder,
                 edit_encoder: SequenceEncoder,
                 output_sequence_decoder: SeqDecoder,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 learn_bidirectional_edits: bool=True) -> None:
        super(CopyEditor, self).__init__(name, hyperparameters)
        self.__input_sequence_encoder = input_sequence_encoder
        self.__edit_encoder = edit_encoder
        self.__output_sequence_decoder = output_sequence_decoder
        self.__learn_reverse_edits = learn_bidirectional_edits

        self.__reverse_edit_layer = None

    def _finalize_component_metadata_and_model(self) -> None:
        if self.__learn_reverse_edits:
            self.__reverse_edit_layer = nn.Linear(
                in_features=self.__edit_encoder.summary_state_size,
                out_features=self.__edit_encoder.summary_state_size
            )

    @property
    def input_sequence_encoder(self):
        return self.__input_sequence_encoder

    @property
    def output_sequence_decoder(self):
        return self.__output_sequence_decoder

    @property
    def edit_encoder(self):
        return self.__edit_encoder

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return { }

    def _load_metadata_from_sample(self, data_to_load: Edit) -> None:
        self.__input_sequence_encoder.load_metadata_from_sample(data_to_load.input_sequence)
        if self.__learn_reverse_edits:
            self.__input_sequence_encoder.load_metadata_from_sample(data_to_load.output_sequence)
        self.__output_sequence_decoder.load_metadata_from_sample(SeqDecoder.InputOutputSequence(
            input_sequence=data_to_load.input_sequence,
            output_sequence=data_to_load.output_sequence
        ))
        if self.__learn_reverse_edits:
            self.__output_sequence_decoder.load_metadata_from_sample(SeqDecoder.InputOutputSequence(
                input_sequence=data_to_load.output_sequence,
                output_sequence=data_to_load.input_sequence,
            ))

        # If the edit encoder is using the same token encoders as input/output then things will be counted more
        # than 1 times
        self.__edit_encoder.load_metadata_from_sample(data_to_load)

    TensorizedData = NamedTuple('CopyEditorTensorizedData', [
        ('input_sequence', Any),
        ('input_sequence_r', Any),
        ('output_sequence', Any),
        ('output_sequence_r', Any),
        ('aligned_edits', Any),
    ])

    def load_data_from_sample(self, data_to_load: Edit) -> Optional['CopyEditor.TensorizedData']:
        return self.TensorizedData(
            input_sequence=self.__input_sequence_encoder.load_data_from_sample(data_to_load.input_sequence),
            input_sequence_r=self.__input_sequence_encoder.load_data_from_sample(data_to_load.output_sequence)
                                                                            if self.__learn_reverse_edits else None,
            output_sequence=self.__output_sequence_decoder.load_data_from_sample(data_to_load),
            output_sequence_r=self.__output_sequence_decoder.load_data_from_sample(SeqDecoder.InputOutputSequence(
                        input_sequence= data_to_load.output_sequence,
                        output_sequence= data_to_load.input_sequence,
                    )) if self.__learn_reverse_edits else None,
            aligned_edits=self.__edit_encoder.load_data_from_sample(data_to_load)
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            'input_sequences': self.__input_sequence_encoder.initialize_minibatch(),
            'input_sequences_r': self.__input_sequence_encoder.initialize_minibatch() if self.__learn_reverse_edits else None,
            'output_sequences': self.__output_sequence_decoder.initialize_minibatch(),
            'output_sequences_r': self.__output_sequence_decoder.initialize_minibatch()  if self.__learn_reverse_edits else None,
            'aligned_edits': self.__edit_encoder.initialize_minibatch()
        }

    def extend_minibatch_by_sample(self, datapoint: 'CopyEditor.TensorizedData', accumulated_minibatch_data: Dict[str, Any]) -> bool:
        continue_extending = self.__input_sequence_encoder.extend_minibatch_by_sample(
            datapoint=datapoint.input_sequence,
            accumulated_minibatch_data=accumulated_minibatch_data['input_sequences'])
        continue_extending &= self.__output_sequence_decoder.extend_minibatch_by_sample(
            datapoint=datapoint.output_sequence,
            accumulated_minibatch_data=accumulated_minibatch_data['output_sequences'])

        if self.__learn_reverse_edits:
            continue_extending &= self.__input_sequence_encoder.extend_minibatch_by_sample(
                datapoint=datapoint.input_sequence_r,
                accumulated_minibatch_data=accumulated_minibatch_data['input_sequences_r'])
            continue_extending &= self.__output_sequence_decoder.extend_minibatch_by_sample(
                datapoint=datapoint.output_sequence_r,
                accumulated_minibatch_data=accumulated_minibatch_data['output_sequences_r'])
        continue_extending &= self.__edit_encoder.extend_minibatch_by_sample(
            datapoint=datapoint.aligned_edits,
            accumulated_minibatch_data=accumulated_minibatch_data['aligned_edits']
        )
        return continue_extending

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'input_sequences': self.__input_sequence_encoder.finalize_minibatch(accumulated_minibatch_data['input_sequences']),
            'input_sequences_r': self.__input_sequence_encoder.finalize_minibatch(accumulated_minibatch_data['input_sequences_r'])  if self.__learn_reverse_edits else None,
            'output_sequences': self.__output_sequence_decoder.finalize_minibatch(accumulated_minibatch_data['output_sequences']),
            'output_sequences_r': self.__output_sequence_decoder.finalize_minibatch(accumulated_minibatch_data['output_sequences_r'])  if self.__learn_reverse_edits else None,
            'aligned_edits': self.__edit_encoder.finalize_minibatch(accumulated_minibatch_data['aligned_edits']),
            'edit_type': None
        }

    def forward(self, *, input_sequences: Dict[str, Any], output_sequences: Dict[str, Any],
                input_sequences_r: Dict[str, Any], output_sequences_r: Dict[str, Any], aligned_edits: Dict[str, Any],
                edit_type: Optional[Dict[str, Any]]):

        input_encoding = self.__input_sequence_encoder.forward(
            input_sequence_data=input_sequences,
            return_embedded_sequence=True
        )
        if self.__learn_reverse_edits:
            input_encoding_r = self.__input_sequence_encoder.forward(
                input_sequence_data=input_sequences_r,
                return_embedded_sequence=True)

        memories, memories_lengths, output_state, input_sequence_token_embeddings = input_encoding
        if self.__learn_reverse_edits:
            memories_r, memories_lengths_r, output_state_r, input_sequence_token_embeddings_r = input_encoding_r

        _, _, edit_representations = self.__edit_encoder.forward(input_sequence_data=aligned_edits)


        initial_state = torch.cat([output_state, edit_representations], dim=-1)

        decoder_loss = self.__output_sequence_decoder.forward(memories=memories, memories_lengths=memories_lengths,
                                                              initial_state=initial_state,
                                                              input_sequence_token_embeddings=input_sequence_token_embeddings,
                                                              additional_decoder_input=edit_representations,
                                                              **output_sequences)
        if self.__learn_reverse_edits:
            reverse_edit_rep = self.__reverse_edit_layer(edit_representations)
            initial_state_r = torch.cat([output_state_r, reverse_edit_rep], dim=-1)
            decoder_loss_r = self.__output_sequence_decoder.forward(memories=memories_r, memories_lengths=memories_lengths_r,
                                                                    initial_state=initial_state_r,
                                                                    input_sequence_token_embeddings=input_sequence_token_embeddings_r,
                                                                    additional_decoder_input=reverse_edit_rep,
                                                                    **output_sequences_r)
        else:
            decoder_loss_r = 0

        decoder_loss = decoder_loss + decoder_loss_r
        return decoder_loss

    def get_edit_representations(self, mb_data):
        with torch.no_grad():
            _, _, edit_representations = self.__edit_encoder.forward(input_sequence_data=mb_data['aligned_edits'])
            return edit_representations

    def greedy_decode(self, input_sequences: Dict[str, Any], aligned_edits: Dict[str, Any],
                      ground_input_sequences: List[List[str]], max_length: int=50,
                      fixed_edit_representations: Optional[torch.Tensor]=None) -> List[Tuple[List[List[str]], List[float]]]:
        with torch.no_grad():
            ground_input_sequences, initial_state, memories, memory_lengths, edit_representations = self.__prepare_decoding(aligned_edits,
                                                                                                      ground_input_sequences,
                                                                                                      input_sequences,
                                                                                                      fixed_edit_representations)

            return self.__output_sequence_decoder.greedy_decode(memories, memory_lengths,
                                                                initial_state=initial_state, max_length=max_length,
                                                                memories_str_representations=ground_input_sequences,
                                                                additional_decoder_input=edit_representations)

    def beam_decode(self, input_sequences: Dict[str, Any], aligned_edits: Dict[str, Any],
                    ground_input_sequences: List[List[str]], max_length: int=50,
                    fixed_edit_representations: Optional[torch.Tensor]=None) -> List[Tuple[List[List[str]], List[float]]]:
        with torch.no_grad():
            ground_input_sequences, initial_state, memories, memory_lengths, edit_representations = self.__prepare_decoding(aligned_edits,
                                                                                                      ground_input_sequences,
                                                                                                      input_sequences,
                                                                                                      fixed_edit_representations)

            return self.__output_sequence_decoder.beam_decode(memories, memory_lengths,
                                                              initial_state=initial_state, max_length=max_length,
                                                              memories_str_representations=ground_input_sequences,
                                                              additional_decoder_input= edit_representations
                                                              )


    def __prepare_decoding(self, aligned_edits, ground_input_sequences, input_sequences,
                           fixed_edit_representations: Optional[torch.Tensor]):
        memories, memory_lengths, output_state = self.__input_sequence_encoder.forward(
            input_sequence_data=input_sequences)
        if fixed_edit_representations is None:
            _, _, edit_representation = self.__edit_encoder.forward(input_sequence_data=aligned_edits)
        else:
            edit_representation = fixed_edit_representations
        initial_state = torch.cat([output_state, edit_representation], dim=-1)
        return ground_input_sequences, initial_state, memories, memory_lengths, edit_representation

    def compute_likelihood(self, *, input_sequences: Dict[str, Any], output_sequences: Dict[str, Any],
                input_sequences_r: Dict[str, Any], output_sequences_r: Dict[str, Any], aligned_edits: Dict[str, Any],
                edit_type: Optional[Dict[str, Any]]):
        with torch.no_grad():
            memories, memories_lengths, output_state = self.__input_sequence_encoder.forward(input_sequence_data=input_sequences)
            _, _, edit_representations = self.__edit_encoder.forward(input_sequence_data=aligned_edits)
            initial_state = torch.cat([output_state, edit_representations], dim=-1)
            return self.__output_sequence_decoder.compute_likelihood(memories=memories,
                                                                memories_lengths=memories_lengths,
                                                                initial_state=initial_state,
                                                                additional_decoder_input=edit_representations,
                                                                **output_sequences)
