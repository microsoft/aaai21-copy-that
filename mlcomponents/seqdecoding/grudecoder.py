from typing import Dict, Any, Optional, List, Tuple

import torch
from torch import nn

from mlcomponents.embeddings import SequenceEmbedder
from . import SeqDecoder
from .luongattention import LuongAttention


class GruDecoder(SeqDecoder):
    def __init__(self, name: str, token_encoder: SequenceEmbedder,
                 standard_attention: Optional[LuongAttention]=None,
                 hyperparameters: Optional[Dict[str, Any]]=None,
                 pre_trained_gru: Optional[nn.GRU] = None,
                 include_summarizing_network: bool = True
                 ) -> None:
        super(GruDecoder, self).__init__(name, token_encoder, hyperparameters)
        self.__output_gru = pre_trained_gru  # type: Optional[nn.GRU]
        self.__standard_attention = standard_attention  # type: Optional[LuongAttention]
        self.__dropout_layer = None  # type: Optional[nn.Dropout]
        self.__cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce=False)

        self.__include_summarizing_network = include_summarizing_network
        self.__summarization_layer = None
        self.reset_metrics()

    @property
    def gru(self) -> nn.GRU:
        return self.__output_gru

    def _finalize_component_metadata_and_model(self) -> None:
        if self.__output_gru is None:
            self.__output_gru = nn.GRU(
                input_size=self.target_token_encoder.embedding_size + self.get_hyperparameter('additional_inputs_size'),
                hidden_size=self.get_hyperparameter('hidden_size'),
                batch_first=True
                )
        else:
            # Make sure that GRU is compatible
            assert self.__output_gru.hidden_size == self.get_hyperparameter('hidden_size')
            assert self.__output_gru.num_layers == self.get_hyperparameter('num_layers')
            assert self.__output_gru.input_size == self.target_token_encoder.embedding_size + self.get_hyperparameter('additional_inputs_size')
            assert self.__output_gru.batch_first

        self.__dropout_layer = nn.Dropout(p=self.get_hyperparameter('dropout_rate'))

        if self.__include_summarizing_network:
            self.__summarization_layer = nn.Linear(
                self.get_hyperparameter('initial_state_size') + self.get_hyperparameter('additional_initial_state_inputs_size'),
                self.get_hyperparameter('hidden_size')
            )
        else:
            assert self.get_hyperparameter('initial_state_size') + self.get_hyperparameter('additional_initial_state_inputs_size') ==\
                self.get_hyperparameter('hidden_size'), 'Initial states sizes do not match.'

        self.__hidden_to_output = nn.Parameter(torch.randn(self.get_hyperparameter('hidden_size'),
                                                           self.target_token_encoder.embedding_size,
                                                          dtype=torch.float, requires_grad=True))

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'dropout_rate': 0.2,
            'hidden_size': 64,
            'initial_state_size': 64,
            'additional_initial_state_inputs_size': 0,
            'additional_inputs_size': 0
        }

    def load_data_from_sample(self, data_to_load: SeqDecoder.InputOutputSequence) -> Optional[Any]:
        return self.target_token_encoder.load_data_from_sample(self._add_start_end(data_to_load.output_sequence))

    def initialize_minibatch(self) -> Dict[str, Any]:
        return self.target_token_encoder.initialize_minibatch()

    def extend_minibatch_by_sample(self, datapoint: Any, accumulated_minibatch_data: Dict[str, Any]) -> bool:
        return self.target_token_encoder.extend_minibatch_by_sample(
            datapoint=datapoint,
            accumulated_minibatch_data=accumulated_minibatch_data)

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        return {'output_sequences': self.target_token_encoder.finalize_minibatch(accumulated_minibatch_data)}

    def __get_output_logits(self, target_token_embeddings, state, memories, memory_lengths):
        """

        :param target_token_embeddings: B x max-out-len x D
        :param state: B x D
        :param memories: B x max-in-len x H
        :param memory_lengths: B
        :return: output_logits: B x max-out-len x V, h_out: num-layers x B x H
        """
        output_states, h_out = self.__output_gru.forward(target_token_embeddings, state.unsqueeze(0))  # B x max-out-len-1 x H
        if self.__standard_attention is not None:
            output_states_flat = self.__standard_attention.forward(
                memories=memories, memories_length=memory_lengths,
                lookup_vectors=output_states.contiguous())  # (B * max-out-len - 1) x H
            output_states = output_states_flat.view(output_states.shape)
        output_logits = torch.einsum('blh,hd,vd->blv', output_states, self.__hidden_to_output,
                                     self.target_token_encoder.embedding_matrix)
        return output_logits, h_out

    def forward(self, *, memories=None, memories_lengths=None, initial_state=None, output_sequences: Dict[str, Any]=None):
        target_token_embeddings, sequence_lengths = self.target_token_encoder.forward(as_packed_sequence=False,
                                                                                       **output_sequences)  # B x max_len x D and B

        if initial_state is None:
            initial_state = torch.zeros((sequence_lengths.shape[0], self.get_hyperparameter('hidden_size')), device=self.device)
        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        target_token_embeddings = target_token_embeddings[:, :-1]
        output_logits, _ = self.__get_output_logits(target_token_embeddings, initial_state, memories, memories_lengths)

        loss = self.__cross_entropy_loss(input=output_logits.flatten(0,1), target=output_sequences['token_ids'][:, 1:].flatten(0, 1))
        mask = torch.arange(target_token_embeddings.shape[1], device=self.device).unsqueeze(0) <= sequence_lengths.unsqueeze(-1)  # B x max_len - 1
        loss = loss.view(output_logits.shape[0], output_logits.shape[1]) * mask.float()
        loss = (loss.sum(dim=-1) / mask.sum(dim=-1).float()).mean()

        with torch.no_grad():
            self._num_minibatches += 1
            self._loss_sum += float(loss)

        return loss

    def compute_likelihood(self, *, memories, memories_lengths, initial_state, additional_decoder_input, **kwargs):
        raise NotImplemented()

    def greedy_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]], max_length: int=10) -> List[Tuple[List[List[str]], List[float]]]:
        vocabulary = self.target_token_encoder.vocabulary

        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)

        current_decoder_state = initial_state  # B x H
        next_token_sequences = torch.tensor(
            [[vocabulary.get_id_or_unk(self.START)]] * initial_state.shape[0],
            device=self.device)  # B x 1

        predicted_tokens = []  # List[torch.Tensor]
        predicted_logprobs = []  # List[float]
        for i in range(max_length):
            # Embed next_token_sequences
            target_token_embeddings, _ = self.target_token_encoder.forward(
                as_packed_sequence=False,
                token_ids=next_token_sequences,
                lengths=None)  # B x 1 x D
            output_logits, current_decoder_state = self.__get_output_logits(target_token_embeddings, current_decoder_state, memories, memory_lengths)
            current_decoder_state = current_decoder_state.squeeze(0)
            output = nn.functional.log_softmax(output_logits, dim=-1)
            greedy_word_logprobs, next_tokens = output.max(dim=-1)

            predicted_logprobs.append(greedy_word_logprobs.squeeze(1))
            predicted_tokens.append(next_tokens.squeeze(1))
            next_token_sequences = next_tokens

        # Now convert sequences back to str
        predictions = []  # type: List[Tuple[List[str], float]]
        for i in range(initial_state.shape[0]):
            tokens = [vocabulary.get_name_for_id(int(predicted_tokens[j][i])) for j in range(max_length)]
            try:
                end_idx = tokens.index(self.END)
            except ValueError:
                end_idx = max_length
            tokens = tokens[:end_idx]
            logprob = sum(float(predicted_logprobs[j][i].cpu()) for j in range(min(end_idx, max_length)))
            predictions.append(([tokens], [logprob]))
        return predictions

    def beam_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                    max_length: int = 40, max_beam_size: int = 5) -> List[Tuple[List[List[str]], List[float]]]:
        raise NotImplementedError
