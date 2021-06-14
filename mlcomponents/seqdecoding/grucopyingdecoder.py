import heapq
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, NamedTuple

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
from torch import nn

from mlcomponents.embeddings import TokenSequenceEmbedder
from . import SeqDecoder
from .luongattention import LuongAttention

LARGE_NUMBER = 5000


class GruCopyingDecoder(SeqDecoder):
    def __init__(self, name: str, token_encoder: TokenSequenceEmbedder,
                 standard_attention: Optional[LuongAttention]=None,
                 pre_trained_gru: Optional[nn.GRU] = None,
                 hyperparameters: Optional[Dict[str, Any]]=None,
                 include_summarizing_network: bool = True
                 ) -> None:
        super(GruCopyingDecoder, self).__init__(name, token_encoder, hyperparameters)
        self.__output_gru = None # type: Optional[nn.GRU]
        self.__standard_attention = standard_attention  # type: Optional[LuongAttention]
        self.__dropout_layer = None # type: Optional[nn.Dropout]
        self.__cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce=False)

        self.__include_summarizing_network = include_summarizing_network
        self.__summarization_layer = None
        self.reset_metrics()

    def _finalize_component_metadata_and_model(self) -> None:
        if self.__output_gru is None:
            self.__output_gru = nn.GRU(
                input_size=self.target_token_encoder.embedding_size + self.get_hyperparameter('additional_inputs_size'),
                hidden_size=self.get_hyperparameter('hidden_size'),
                num_layers=self.get_hyperparameter('num_layers'),
                batch_first=True
            )
        else:
            assert self.__output_gru.hidden_size == self.get_hyperparameter('hidden_size')
            assert self.__output_gru.num_layers == 1
            assert self.__output_gru.input_size == self.target_token_encoder.embedding_size + self.get_hyperparameter('additional_inputs_size')
            assert self.__output_gru.batch_first

        self.__dropout_layer = nn.Dropout(p=self.get_hyperparameter('dropout_rate'))

        if self.__include_summarizing_network:
            self.__summarization_layer = nn.Linear(
                self.get_hyperparameter('initial_state_size') + self.get_hyperparameter('additional_initial_state_inputs_size'),
                self.get_hyperparameter('hidden_size') * self.get_hyperparameter('num_layers'), bias=False
            )
        else:
            assert self.get_hyperparameter('initial_state_size') + self.get_hyperparameter('additional_initial_state_inputs_size') ==\
                self.get_hyperparameter('hidden_size') * self.get_hyperparameter('num_layers'), 'Initial states sizes do not match.'

        self.__hidden_to_output = nn.Parameter(torch.randn(self.get_hyperparameter('hidden_size'),
                                                           self.target_token_encoder.embedding_size,
                                                          dtype=torch.float, requires_grad=True)*.1)
        self.__hidden_to_query_vector = nn.Parameter(torch.randn(self.get_hyperparameter('memories_hidden_dimension'),
                                                           self.get_hyperparameter('hidden_size'),
                                                           dtype=torch.float, requires_grad=True)*.1)
        self.__vocabulary_bias = nn.Parameter(torch.zeros(len(self.target_token_encoder.vocabulary),
                                                          dtype=torch.float, requires_grad=True))

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'dropout_rate': 0.2,
            'hidden_size': 64,
            'num_layers': 1,
            'initial_state_size': 64,
            'memories_hidden_dimension': 128,
            'additional_initial_state_inputs_size': 0,
            'additional_inputs_size': 0,
            'max_memories_length': 25,
            'num_layers': 1,
            'data_as_demonstrator_rate': 0.
        }

    TensorizedData = NamedTuple('GruCopyingDecoderTensorizedData', [
         ('output_sequence', Any),
         ('copy_locations', np.ndarray)
    ])

    def load_data_from_sample(self, data_to_load: SeqDecoder.InputOutputSequence) -> Optional['GruCopyingDecoder.TensorizedData']:
        return self.TensorizedData(
            output_sequence=self.target_token_encoder.load_data_from_sample(self._add_start_end(data_to_load.output_sequence)),
            copy_locations=self.__get_copy_locations(data_to_load.input_sequence, self._add_start_end(data_to_load.output_sequence))
        )

    def __get_copy_locations(self, input_sequence, output_sequence):
        max_in_length = min(len(input_sequence), self.get_hyperparameter('max_memories_length'))
        max_out_length = min(len(output_sequence), self.target_token_encoder.get_hyperparameter('max_seq_length'))

        copy_locations = np.zeros((max_out_length, max_in_length), dtype=np.bool)
        input_sequence_elements = np.array([input_sequence[:max_in_length]], dtype=np.object)  # 1 x I
        output_sequence_elements = np.array([output_sequence[:max_out_length]], dtype=np.object).T  # O x 1
        copy_locations[:len(output_sequence), :len(input_sequence)] = input_sequence_elements == output_sequence_elements  # O x I
        return copy_locations

    def initialize_minibatch(self) -> Dict[str, Any]:
        return  {
            'output_sequences': self.target_token_encoder.initialize_minibatch(),
            'copy_locations': []
        }

    def extend_minibatch_by_sample(self, datapoint: 'GruCopyingDecoder.TensorizedData', accumulated_minibatch_data: Dict[str, Any]) -> bool:
        continue_extending = self.target_token_encoder.extend_minibatch_by_sample(
                                    datapoint=datapoint.output_sequence,
                                    accumulated_minibatch_data=accumulated_minibatch_data['output_sequences'])
        accumulated_minibatch_data['copy_locations'].append(datapoint.copy_locations)
        return continue_extending

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        accumulated_copy_locations = accumulated_minibatch_data['copy_locations']
        max_in_length = max(t.shape[1] for t in accumulated_copy_locations)
        max_out_length = max(t.shape[0] for t in accumulated_copy_locations)
        batch_size = len(accumulated_copy_locations)

        copy_locations = np.zeros((batch_size, max_out_length, max_in_length), dtype=np.bool)
        for i in range(batch_size):
            locations_for_i = accumulated_copy_locations[i]
            copy_locations[i, :locations_for_i.shape[0], :locations_for_i.shape[1]] = locations_for_i

        return {
            'output_sequences': self.target_token_encoder.finalize_minibatch(accumulated_minibatch_data['output_sequences']),
            'copy_locations': torch.tensor(copy_locations, dtype=torch.int64, device=self.device)
        }


    def __get_output_logits(self, decoded_token_embeddings, state, memories, memory_lengths, sample_rate: float=0,
                            input_sequence_token_embeddings=None, additional_rnn_input=None):
        if additional_rnn_input is not None:
            tiled_additional_decoder_input = additional_rnn_input.unsqueeze(1).repeat(1,
                                                                                  decoded_token_embeddings.size(1),
                                                                                  1)  # B x max-out-len-1 x D
            tiled_additional_decoder_input = torch.repeat_interleave(tiled_additional_decoder_input,
                                                                 int(decoded_token_embeddings.size(
                                                                     0) / additional_rnn_input.size(0)),
                                                                 dim=0)  # B*beam width x max-out-len-1 x D
            decoded_token_embeddings = torch.cat([decoded_token_embeddings, tiled_additional_decoder_input], dim=-1)

        if not self.training or sample_rate == 0 or input_sequence_token_embeddings is None:
            return self.__get_output_logits_xent(decoded_token_embeddings, state, memories, memory_lengths)
        else:
            if input_sequence_token_embeddings is None:
                raise Exception('The input sequence token embeddings cannot be None when using data-as-demonstrator.')
            return self.__get_output_logits_dad(decoded_token_embeddings, state, memories, memory_lengths, sample_rate,
                                                input_sequence_token_embeddings, additional_rnn_input)

    def __get_output_logits_xent(self, decoded_token_embeddings, state, memories, memory_lengths):
        """

        :param decoded_token_embeddings: B x max-out-len x D
        :param state: B x D
        :param memories: B x max-in-len x H
        :param memory_lengths: B
        :return: output_logits: B x max-out-len x V, copy_logits B x max-out-len x max-in-len, h_out: num-layers x B x H
        """
        output_states, h_out = self.__output_gru.forward(decoded_token_embeddings, state)  # B x max-out-len-1 x H
        output_states = output_states.contiguous()
        if self.__standard_attention is not None:
            output_states_flat = self.__standard_attention.forward(
                memories=memories, memories_length=memory_lengths,
                lookup_vectors=output_states)  # (B * max-out-len - 1) x H
            output_states = output_states_flat.view(output_states.shape)

        output_logits = torch.einsum('blh,hd,vd->blv', output_states, self.__hidden_to_output,
                                     self.target_token_encoder.embedding_matrix) + self.__vocabulary_bias.view(1, 1, -1)

        copy_logits = torch.einsum('bmh,hd,bld->blm', memories, self.__hidden_to_query_vector, output_states)

        copy_logits.masked_fill_(mask=torch.arange(memories.shape[1], device=self.device).view(1, 1, -1) >= memory_lengths.view(-1, 1, 1),
                                value=float('-inf'))
        return output_logits, copy_logits, h_out

    def __get_output_logits_dad(self, decoded_token_embeddings, state, memories, memory_lengths,
                                sample_rate: float, input_sequence_token_embeddings, rnn_additional_input):
        """

        :param decoded_token_embeddings: B x max-out-len x D
        :param state: num_layers x B x D
        :param memories: B x max-in-len x H
        :param memory_lengths: B
        :return: output_logits: B x max-out-len x V, copy_logits B x max-out-len x max-in-len, h_out: num-layers x B x H
        """
        output_logits = []
        copy_logits = []
        current_state = state  # num_layers x B x D

        for i in range(decoded_token_embeddings.shape[1]):
            # Sample inputs
            use_base = np.random.random() > sample_rate
            if use_base or i == 0:
                inputs = decoded_token_embeddings[:, i]
            else:
                # Sample from previous decision
                last_prediction_logits = torch.cat([output_logits[-1], copy_logits[-1]], dim=-1)  # B x V + max-len
                sampled_ids = torch.multinomial(nn.functional.softmax(last_prediction_logits, dim=-1), num_samples=1).squeeze(1)  # B

                # To avoid constructing a large batched lookup table, do this separately.
                vocab_embeddings = self.target_token_encoder.embedding_matrix[
                    torch.min(sampled_ids,
                              torch.ones_like(sampled_ids) * self.target_token_encoder.embedding_matrix.shape[0]-1)]
                copy_embeddings = input_sequence_token_embeddings[
                    torch.arange(sampled_ids.shape[0], device=self.device),
                    torch.nn.functional.relu(sampled_ids-self.target_token_encoder.embedding_matrix.shape[0]) # Clamp to 0
                ]
                inputs = torch.where((sampled_ids<self.target_token_encoder.embedding_matrix.shape[0]).view(-1, 1),
                                                   vocab_embeddings,
                                                   copy_embeddings) # B x D
                if rnn_additional_input is not None:
                    inputs = torch.cat([inputs, rnn_additional_input], dim=-1)

            # Now run one step of GRU
            inputs = inputs.unsqueeze(1)  # B x 1 x D
            output_states, h_out = self.__output_gru.forward(inputs,
                                                             current_state)
            output_states = output_states.contiguous()
            if self.__standard_attention is not None:
                output_states_flat = self.__standard_attention.forward(
                    memories=memories, memories_length=memory_lengths,
                    lookup_vectors=output_states)  # (B * 1) x H
                output_states = output_states_flat.view(output_states.shape)
            output_states = output_states.squeeze(1) # B x H

            next_output_logits = torch.einsum('bh,hd,vd->bv', output_states, self.__hidden_to_output,
                                         self.target_token_encoder.embedding_matrix) + self.__vocabulary_bias.view(1, -1)

            next_copy_logits = torch.einsum('bmh,hd,bd->bm', memories, self.__hidden_to_query_vector, output_states)
            next_copy_logits.masked_fill_(torch.arange(next_copy_logits.shape[1], device=self.device).view(1, -1) >= memory_lengths.view(-1, 1), float('-inf'))

            # Append logits
            output_logits.append(next_output_logits)
            copy_logits.append(next_copy_logits)
            current_state = h_out
        return torch.stack(output_logits, dim=1), torch.stack(copy_logits, dim=1), current_state


    def forward(self, *, memories, memories_lengths, initial_state, copy_locations, output_sequences: Dict[str, Any],
                input_sequence_token_embeddings=None, additional_decoder_input=None):
        likelihood = self.compute_likelihood(memories=memories, memories_lengths=memories_lengths,
                                             initial_state=initial_state, copy_locations=copy_locations, output_sequences=output_sequences,
                                             input_sequence_token_embeddings=input_sequence_token_embeddings,
                                             additional_decoder_input=additional_decoder_input, normalize=True)
        loss = -(likelihood).mean()

        with torch.no_grad():
            self._num_minibatches += 1
            self._loss_sum += float(loss)

        return loss

    def compute_likelihood(self, *, memories, memories_lengths, initial_state, copy_locations, output_sequences: Dict[str, Any],
                           input_sequence_token_embeddings=None, normalize: bool=False, additional_decoder_input=None,
                           return_debug_info: bool = False):
        # copy_locations: B x max-out-len x max-inp-len
        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        initial_state = torch.reshape(initial_state,
            (initial_state.shape[0], self.get_hyperparameter('num_layers'), self.get_hyperparameter('hidden_size'))).transpose(0, 1).contiguous() # num_layers x B x H

        copy_locations = copy_locations[:, 1:, :memories.shape[1]]  # The input might be truncated if all sequences are smaller.

        target_token_embeddings, sequence_lengths = self.target_token_encoder.forward(as_packed_sequence=False,
                                                                                       **output_sequences)  # B x max_len x D and B

        target_token_embeddings = target_token_embeddings[:, :-1]

        output_logits, copy_logits, _ = self.__get_output_logits(target_token_embeddings, initial_state, memories, memories_lengths,
                                                                 sample_rate=self.get_hyperparameter('data_as_demonstrator_rate') if self.training else 0,
                                                                 input_sequence_token_embeddings=input_sequence_token_embeddings,
                                                                 additional_rnn_input=additional_decoder_input
                                                                 )

        # Merge the output and copy logits
        logits = torch.cat([output_logits, copy_logits], dim=-1)  # B x max-out-len x V + max-in-len
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        can_copy = copy_locations.max(dim=-1)[0].gt(0)  # B x max-out-len

        target_generated_tokens = output_sequences['token_ids'][:, 1:].flatten(0, 1)
        target_is_unk = target_generated_tokens.eq(self.target_token_encoder.vocabulary.get_id_or_unk(Vocabulary.get_unk()))
        generation_logprobs = log_probs.flatten(0, 1)[torch.arange(target_generated_tokens.shape[0]), target_generated_tokens]
        generation_logprobs = generation_logprobs + (target_is_unk & can_copy.flatten(0, 1)).float() * -LARGE_NUMBER # B x max-out-len

        copy_logprobs = log_probs[:, :, -copy_logits.shape[-1]:]  # B x max-out-len x max-in-len
        copy_logprobs = copy_logprobs + copy_locations.ne(1).float() * -LARGE_NUMBER
        copy_logprobs = copy_logprobs.logsumexp(dim=-1)  # B x max-out-len

        mask = torch.arange(target_token_embeddings.shape[1], device=self.device).unsqueeze(0) < sequence_lengths.unsqueeze(-1)-1 # B x max-out-len - 1
        full_logprob = torch.logsumexp(torch.stack([generation_logprobs, copy_logprobs.flatten(0, 1)], dim=-1), dim=-1)  # B x max-out-len
        full_logprob = full_logprob.view(copy_logprobs.shape) * mask.float()
        if normalize:
            return full_logprob.sum(dim=-1) / (sequence_lengths-1).float()
        else:
            return full_logprob.sum(dim=-1)

    def greedy_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                      max_length: int=40, additional_decoder_input=None) -> List[Tuple[List[List[str]], List[float]]]:
        vocabulary = self.target_token_encoder.vocabulary

        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        initial_state = torch.reshape(initial_state, (self.get_hyperparameter('num_layers'), -1, self.get_hyperparameter('hidden_size'))) # num_layers x B x H

        current_decoder_state = initial_state  # num_layers x B x H
        next_token_sequences = torch.tensor(
            [[vocabulary.get_id_or_unk(self.START)]] * memories.shape[0],
            device=self.device)  # B x 1

        predicted_tokens = []  # List[List[str]]
        predicted_logprobs = []  # List[float]
        for i in range(max_length):
            # Embed next_token_sequences
            target_token_embeddings, _ = self.target_token_encoder.forward(
                as_packed_sequence=False,
                token_ids=next_token_sequences,
                lengths=None)  # B x 1 x D
            output_logits, copy_logits, current_decoder_state = self.__get_output_logits(
                target_token_embeddings, current_decoder_state, memories, memory_lengths, additional_rnn_input=additional_decoder_input)

            logits = torch.cat([output_logits, copy_logits], dim=-1)  # B x 1 x V + max-in-len
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            copy_logprobs = log_probs[:, :, -copy_logits.shape[-1]:]

            log_probs = log_probs.cpu().numpy()
            copy_logprobs = copy_logprobs.cpu().numpy()

            output_logprobs = []  # type: List[Dict[str, float]]
            for j in range(logits.shape[0]):
                sample_logprobs = defaultdict(lambda :float('-inf'))
                for k in range(len(vocabulary)):
                    sample_logprobs[vocabulary.get_name_for_id(k)] = log_probs[j, 0, k]
                output_logprobs.append(sample_logprobs)

            for j in range(logits.shape[0]):
                for k in range(memory_lengths[j]):
                    target_word = memories_str_representations[j][k]
                    output_logprobs[j][target_word] = np.logaddexp(output_logprobs[j][target_word], copy_logprobs[j, 0, k])

            predicted_tokens_for_this_step = []
            predicted_logprobs_for_this_step = []
            for j in range(logits.shape[0]):
                best_word, best_logprob = None, float('-inf')
                for word, logprob in output_logprobs[j].items():
                    if best_logprob < logprob:
                        best_logprob = logprob
                        best_word = word
                assert best_word is not None
                predicted_tokens_for_this_step.append(best_word)
                predicted_logprobs_for_this_step.append(best_logprob)


            predicted_logprobs.append(predicted_logprobs_for_this_step)
            predicted_tokens.append(predicted_tokens_for_this_step)
            next_token_sequences = torch.tensor([[vocabulary.get_id_or_unk(t)] for t in predicted_tokens_for_this_step],
                                                device=self.device)

        return self.convert_ids_to_str(initial_state.shape[0], max_length, predicted_logprobs, predicted_tokens)

    def convert_ids_to_str(self, batch_size: int, max_length: int, predicted_logprobs, predicted_tokens):
        predictions = []  # type: List[Tuple[List[List[str]], List[float]]]
        for i in range(batch_size):
            tokens = [predicted_tokens[j][i] for j in range(max_length)]
            try:
                end_idx = tokens.index(self.END)
            except ValueError:
                end_idx = max_length
            tokens = tokens[:end_idx]
            logprob = sum(float(predicted_logprobs[j][i]) for j in range(min(end_idx, max_length)))
            predictions.append(([tokens], [logprob]))
        return predictions

    def beam_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                    max_length: int=150, max_beam_size: int=20, additional_decoder_input=None) -> List[Tuple[List[List[str]], List[float]]]:
        vocabulary = self.target_token_encoder.vocabulary

        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        initial_state = torch.reshape(initial_state, (self.get_hyperparameter('num_layers'), -1, self.get_hyperparameter('hidden_size'))) # num_layers x B x H

        batch_size = memory_lengths.shape[0]

        current_decoder_state = initial_state  # B*beam_size=1 x H
        frontier_tokens = torch.tensor(
            [[vocabulary.get_id_or_unk(self.START)]] * batch_size,
            device=self.device).unsqueeze(-1)  # B x beam_size=1 x 1

        sequence_logprobs = torch.zeros(frontier_tokens.shape[:2], dtype=torch.float, device=self.device)
        is_done = np.zeros(frontier_tokens.shape[:2], dtype=np.bool)

        predicted_tokens_beam = [[list() for _ in range(max_beam_size)] for _ in range(batch_size)]  # type: List[List[List[str]]]

        for i in range(max_length):
            if np.all(is_done):
                break

            beam_size = frontier_tokens.shape[1]

            # Embed frontier_tokens
            target_token_embeddings, _ = self.target_token_encoder.forward(
                as_packed_sequence=False,
                token_ids=frontier_tokens.flatten(0, 1),
                lengths=None)  # B*beam_size x 1 x D
            output_logits, copy_logits, current_decoder_state = self.__get_output_logits(
                target_token_embeddings, current_decoder_state,
                memories.unsqueeze(1).expand(-1, beam_size, -1, -1).flatten(0,1),
                memory_lengths.unsqueeze(1).expand(-1, beam_size).flatten(0,1), additional_rnn_input=additional_decoder_input)
            current_decoder_state = current_decoder_state.transpose(0, 1).contiguous() # B*beam_size x num_layers x H

            logits = torch.cat([output_logits, copy_logits], dim=-1)  # B*beam_size x 1 x V + max-in-len
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            copy_logprobs = log_probs[:, :, -copy_logits.shape[-1]:]

            log_probs = log_probs.cpu().numpy()
            copy_logprobs = copy_logprobs.cpu().numpy()

            next_sequence_logprobs = torch.zeros((batch_size, max_beam_size), dtype=torch.float, device=self.device)
            next_is_done = np.zeros((batch_size, max_beam_size), dtype=np.bool)
            next_frontier_tokens = torch.zeros((batch_size, max_beam_size), dtype=torch.int64, device=self.device)
            next_decoder_state = torch.zeros((batch_size * max_beam_size, self.get_hyperparameter('num_layers'), current_decoder_state.shape[-1]),
                                                 dtype=torch.float, device=self.device)

            for batch_idx in range(batch_size):
                per_beam_logprobs = []
                for beam_idx in range(beam_size):
                    idx = batch_idx * frontier_tokens.shape[1] + beam_idx

                    sample_logprobs = defaultdict(lambda: float('-inf'))
                    for k in np.argsort(-log_probs[idx, 0, :len(vocabulary)])[:200]:  # To speed things up use only top words
                        sample_logprobs[vocabulary.id_to_token[k]] = log_probs[idx, 0, k]

                    for k in range(memory_lengths[batch_idx]):
                        target_word = memories_str_representations[batch_idx][k]
                        if target_word in sample_logprobs:
                            sample_logprobs[target_word] = np.logaddexp(sample_logprobs[target_word], copy_logprobs[idx, 0, k])
                        else:
                            sample_logprobs[target_word] = copy_logprobs[idx, 0, k]
                    per_beam_logprobs.append(sample_logprobs)

                # Pick next beam
                def all_elements():
                    for beam_idx in range(beam_size):
                        if is_done[batch_idx][beam_idx]:
                            yield beam_idx, None, True, sequence_logprobs[batch_idx, beam_idx]
                        else:
                            for word, word_logprob in per_beam_logprobs[beam_idx].items():
                                yield beam_idx, word, word == self.END, word_logprob + sequence_logprobs[batch_idx, beam_idx]

                top_elements = heapq.nlargest(n=max_beam_size, iterable=all_elements(), key=lambda x:x[-1])
                old_beam = predicted_tokens_beam[batch_idx]
                new_beam = [list() for _ in range(max_beam_size)]
                for i, (beam_idx, word, beam_is_done, seq_logprob) in enumerate(top_elements):
                    next_frontier_tokens[batch_idx, i] = vocabulary.get_id_or_unk(word)
                    next_is_done[batch_idx, i] = beam_is_done
                    next_sequence_logprobs[batch_idx, i] = float(seq_logprob)
                    if beam_is_done:
                        new_beam[i] = old_beam[beam_idx]
                    else:
                        new_beam[i] = old_beam[beam_idx] + [ word ]
                    next_decoder_state[batch_idx * max_beam_size + i, :] = current_decoder_state[batch_idx * beam_size + beam_idx]

                predicted_tokens_beam[batch_idx] = new_beam


            # After we are done for all batches
            is_done = next_is_done
            sequence_logprobs = next_sequence_logprobs.cpu().numpy()
            frontier_tokens = next_frontier_tokens.unsqueeze(-1)
            current_decoder_state = next_decoder_state.transpose(0, 1).contiguous()

        return [(predicted_tokens_beam[i], sequence_logprobs[i]) for i in range(batch_size)]
