from typing import Optional, Dict, Any, NamedTuple, List, Tuple

import numpy as np
import torch
from dpu_utils.mlutils import Vocabulary
import torch
from torch import nn

from data.spanutils import get_copyable_spans
from mlcomponents.embeddings import TokenSequenceEmbedder
from mlcomponents.seqdecoding import SeqDecoder, LuongAttention

BIG_NUMBER = 100

class GruSpanCopyingDecoder(SeqDecoder):
    def __init__(self, name: str, token_encoder: TokenSequenceEmbedder,
                 standard_attention: Optional[LuongAttention]=None,
                 copy_attention: Optional[LuongAttention]=None,
                 hyperparameters: Optional[Dict[str, Any]]=None,
                 include_summarizing_network: bool = True,
                 pre_trained_gru: Optional[nn.GRU] = None
                 ) -> None:
        super(GruSpanCopyingDecoder, self).__init__(name, token_encoder, hyperparameters)
        self.__output_gru = pre_trained_gru # type: Optional[nn.GRU]
        self.__standard_attention = standard_attention  # type: Optional[LuongAttention]
        self.__copy_attention = copy_attention  # type: Optional[LuongAttention]
        self.__dropout_layer = None # type: Optional[nn.Dropout]

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
            # Make sure that GRU is compatible
            assert self.__output_gru.hidden_size == self.get_hyperparameter('hidden_size')
            assert self.__output_gru.num_layers == self.get_hyperparameter('num_layers')
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
                                                          dtype=torch.float, requires_grad=True))

        k = 3 if self.get_hyperparameter('use_max_pool_span_repr') else 2

        self.__hidden_to_end_span_query_vector = nn.Parameter(
            torch.randn(k*self.get_hyperparameter('memories_hidden_dimension'),
                        self.get_hyperparameter('hidden_size'),
                        dtype=torch.float, requires_grad=True))
        self.__vocabulary_bias = nn.Parameter(torch.zeros(len(self.target_token_encoder.vocabulary),
                                                          dtype=torch.float, requires_grad=True))

        # A constant array with the relative span-lengths
        span_lengths = np.zeros((self.get_hyperparameter('max_memories_length'), self.get_hyperparameter('max_memories_length')), dtype=np.int32)
        for i in range(self.get_hyperparameter('max_memories_length')):
            for j in range(i, self.get_hyperparameter('max_memories_length')):
                span_lengths[i, j] = j - i + 1
        self.__span_lengths = span_lengths

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'dropout_rate': 0.2,
            'hidden_size': 64,
            'initial_state_size': 64,
            'num_layers': 1,
            'memories_hidden_dimension': 128,
            'additional_initial_state_inputs_size': 0,
            'additional_inputs_size': 0,
            'max_memories_length': 25,

            # Ablations for the model
            'use_max_pool_span_repr': False,  # Default: False
            'marginalize_over_copying_decisions': True,  # Default: True
            'teacher_force_longest': False   # Default: False
        }

    TensorizedData = NamedTuple('GruSpanCopyingDecoderTensorizedData', [
        ('output_sequence', Any),
        ('copy_spans', np.ndarray)
    ])

    def load_data_from_sample(self, data_to_load: SeqDecoder.InputOutputSequence) -> Optional['GruSpanCopyingDecoder.TensorizedData']:
        max_seq_len = self.target_token_encoder.get_hyperparameter('max_seq_length')
        target_output_sequence = self._add_start_end(data_to_load.output_sequence)[:max_seq_len]
        copyable_spans = get_copyable_spans(data_to_load.input_sequence[:max_seq_len], target_output_sequence)
        if self.get_hyperparameter('teacher_force_longest'):
            teacher_forced_spans = np.zeros_like(copyable_spans)
            for k in range(copyable_spans.shape[0]):
                copyable_ranges = np.nonzero(copyable_spans[k])
                if len(copyable_ranges[0]) == 0:
                    continue
                max_i, max_j = max(zip(*copyable_ranges), key=lambda x: x[1]- x[0])
                teacher_forced_spans[k, max_i, max_j] = True
            copyable_spans = teacher_forced_spans
        return self.TensorizedData(
            output_sequence=self.target_token_encoder.load_data_from_sample(target_output_sequence),
            copy_spans=copyable_spans
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return  {
            'output_sequences': self.target_token_encoder.initialize_minibatch(),
            'copy_spans': []
        }

    def extend_minibatch_by_sample(self, datapoint: 'GruSpanCopyingDecoder.TensorizedData', accumulated_minibatch_data: Dict[str, Any]) -> bool:
        continue_extending = self.target_token_encoder.extend_minibatch_by_sample(
                                    datapoint=datapoint.output_sequence,
                                    accumulated_minibatch_data=accumulated_minibatch_data['output_sequences'])
        accumulated_minibatch_data['copy_spans'].append(datapoint.copy_spans)
        return continue_extending

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        accumulated_copy_spans = accumulated_minibatch_data['copy_spans']
        max_out_length = min(self.target_token_encoder.get_hyperparameter('max_seq_length'), max(c.shape[0] for c in accumulated_copy_spans))
        max_in_length = min(self.get_hyperparameter('max_memories_length'), max(c.shape[1] for c in accumulated_copy_spans))
        padded_copy_spans = np.zeros((len(accumulated_copy_spans), max_out_length-1, max_in_length, max_in_length), dtype=np.bool)

        for i, copy_span in enumerate(accumulated_copy_spans):
            copy_spans = accumulated_copy_spans[i]
            out_seq_size = min(max_out_length, copy_spans.shape[0])
            inp_seq_size = min(max_in_length, copy_spans.shape[1])
            # Start from 1 because of <sos>
            padded_copy_spans[i, :out_seq_size-1, :inp_seq_size, :inp_seq_size] = copy_spans[1:out_seq_size, :inp_seq_size, :inp_seq_size]

        return {
            'output_sequences': self.target_token_encoder.finalize_minibatch(accumulated_minibatch_data['output_sequences']),
            'copyable_spans': torch.tensor(padded_copy_spans, dtype=torch.uint8, device=self.device)
        }

    def forward(self, *, memories, memories_lengths, initial_state, copyable_spans, output_sequences: Dict[str, Any],
                input_sequence_token_embeddings=None, additional_decoder_input=None, **kwargs):
        likelihood = self.compute_likelihood(memories=memories, memories_lengths=memories_lengths, initial_state=initial_state,
                                             copyable_spans=copyable_spans, output_sequences=output_sequences,
                                             input_sequence_token_embeddings=input_sequence_token_embeddings,
                                             additional_decoder_input=additional_decoder_input, normalize=True)

        loss = -likelihood.mean()

        with torch.no_grad():
            self._num_minibatches += 1
            self._loss_sum += float(loss)

        return loss

    def __get_output_logprobs(self, decoded_token_embeddings, state, memories, memory_lengths, additional_decoder_input=None):
        if additional_decoder_input is not None:
            tiled_inputs = additional_decoder_input.unsqueeze(1).repeat(1, decoded_token_embeddings.size(1), 1)  # B x max-out-len-1 x D
            decoded_token_embeddings = torch.cat([decoded_token_embeddings, tiled_inputs], dim=-1)  # B x max-out-len-1 x 2*D

        output_states, h_out = self.__output_gru.forward(decoded_token_embeddings,
                                                         state)  # B x max-out-len-1 x H
        output_states = output_states.contiguous()

        output_states_flat = self.__standard_attention.forward(
            memories=memories, memories_length=memory_lengths,
            lookup_vectors=output_states)  # (B * max-out-len - 1) x H
        output_states_w_attention = output_states_flat.view(output_states.shape)

        output_logits = torch.einsum('blh,hd,vd->blv', output_states_w_attention, self.__hidden_to_output,
                                     self.target_token_encoder.embedding_matrix) + self.__vocabulary_bias.unsqueeze(0).unsqueeze(0)

        copy_output_states_flat = self.__copy_attention.forward(
            memories=memories, memories_length=memory_lengths,
            lookup_vectors=output_states)  # (B * max_out_len - 1) x H, B x max_out_len - 1 x max_inp_len
        copy_output_states = copy_output_states_flat.view(output_states.shape)  # B x (max_out_len - 1) x H

        memory_a = memories.unsqueeze(1).repeat_interleave(repeats=memories.shape[1], dim=1)  # B x max-in-len x max-in-len x H
        memory_b = memories.unsqueeze(2).repeat_interleave(repeats=memories.shape[1], dim=2)  # B x max-in-len x max-in-len x H

        all_memory_pairs = [memory_a, memory_b]

        if self.get_hyperparameter('use_max_pool_span_repr'):
            # An additional representation of the span by max pooling the encoder states.
            max_input_length = memories.shape[1]
            all_memories = memories.reshape((memories.shape[0], 1, 1, max_input_length, memories.shape[2]))  # B x 1 x 1 x max-in-len x H
            all_memories = all_memories.repeat((1, max_input_length, max_input_length, 1, 1)) # B x max-in-len x max-in-len x max-in-len x H
            range = torch.arange(max_input_length, device=self.device)

            # valid_range_elements[i,j,k] = True if i <=j and i<=k<=j
            valid_range_elements = (range.reshape(-1, 1, 1) <= range.reshape(1, 1, -1)) & (range.reshape(1, 1, -1) <= range.reshape(1, -1, 1))  # max-in-len x max-in-len x max-in-len
            invalid_range_elements = ~valid_range_elements  # max-in-len x max-in-len x max-in-len

            all_memories.masked_fill_(mask=invalid_range_elements.unsqueeze(0).unsqueeze(-1), value=0)
            span_representation2, _ = all_memories.max(dim=-2)  # B x max-in-len x max-in-len x H

            all_memory_pairs.append(span_representation2)

        all_memory_pairs = torch.cat(all_memory_pairs, dim=-1)  # B x max-in-len x max-in-len x 3H

        copy_span_logits = torch.einsum('bmnh,hd,bld->blmn', all_memory_pairs, self.__hidden_to_end_span_query_vector,
                                         copy_output_states)

        # Fill lower-triangular with -inf (end of span should not be before start) and removed padded memories length
        range = torch.arange(copy_span_logits.shape[-1], device=self.device)
        copy_span_logits.masked_fill_(
            mask=
                range.unsqueeze(-1).gt(range.unsqueeze(0)).unsqueeze(0).unsqueeze(0) |
                torch.arange(memories.shape[1], device=self.device).unsqueeze(0).ge(memory_lengths.unsqueeze(-1)).unsqueeze(1).unsqueeze(-1),  # B x max_out_len - 1 x max_in_len x max_in_len
            value=float('-inf')
        )

        all_action_logits = torch.cat([output_logits, copy_span_logits.flatten(-2)], dim=-1)  # B x max_out_len-1 x V + max_in_len*max_in_len
        all_action_logprobs = nn.functional.log_softmax(all_action_logits, dim=-1)

        generation_logprobs = all_action_logprobs[:, :, :output_logits.shape[-1]]  # B x max_out_len-1 x V
        copy_span_logprobs = all_action_logprobs[:, :, output_logits.shape[-1]:].reshape(copy_span_logits.shape)
        return generation_logprobs, copy_span_logprobs, h_out

    def compute_likelihood(self, *, memories, memories_lengths, initial_state, copyable_spans, output_sequences: Dict[str, Any],
                           input_sequence_token_embeddings=None, data_as_demonstrator_rate: float=0, normalize: bool=False, additional_decoder_input=None,
                           return_debug_info: bool = False):
        # copyable_spans: B x max_out_len-1 x max-inp-len x max-inp-len
        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        initial_state = torch.reshape(initial_state,
            (memories.shape[0], self.get_hyperparameter('num_layers'), self.get_hyperparameter('hidden_size'))).transpose(0, 1).contiguous() # num_layers x B x H
        target_token_embeddings, sequence_lengths = self.target_token_encoder.forward(as_packed_sequence=False,
                                                                                    **output_sequences)  # B x max_len x D and B

        target_token_embeddings = target_token_embeddings[:, :-1]
        generation_logprobs, copy_span_logprobs, _ = self.__get_output_logprobs(target_token_embeddings, initial_state,
                                                                                memories, memories_lengths,
                                                                                additional_decoder_input=additional_decoder_input)

        target_generated_tokens = output_sequences['token_ids'][:, 1:]
        generation_logprobs_flat = generation_logprobs.flatten(0, 1)
        generation_target_logprob = generation_logprobs_flat[
            torch.arange(generation_logprobs_flat.shape[0]),
            target_generated_tokens.flatten(0, 1)
        ].view(target_generated_tokens.shape)  # B x max_out_len - 1

        # Reward an UNK only when we cannot copy
        can_copy = torch.flatten(copyable_spans, start_dim=-2).max(dim=-1)[0]  # B x max_out_len - 1
        target_is_unk = target_generated_tokens.eq(
            self.target_token_encoder.vocabulary.get_id_or_unk(Vocabulary.get_unk()))
        generation_target_logprob.masked_fill_(
            mask=can_copy.bool() & target_is_unk,
            value=float('-inf')
        )

        # Marginalize over all actions
        if self.get_hyperparameter('marginalize_over_copying_decisions'):
            logprob_at_position = torch.zeros_like(can_copy, dtype=torch.float32)  #  B x max_out_len-1
            max_in_length = copy_span_logprobs.shape[-1]
            max_out_length = logprob_at_position.shape[1]
            for i in range(max_out_length-1, -1, -1):
                copy_end_idxs = torch.tensor(i + self.__span_lengths[:max_in_length, :max_in_length], dtype=torch.int64, device=self.device)\
                                    .clamp(0, max_out_length-1)\
                                    .expand(logprob_at_position.shape[0], -1, -1)  # B x max_in_len x max_in_len  # TODO: This is const[i], precompute instead of __span_legths

                generation_after_copy_logprob = logprob_at_position.gather(dim=1, index=copy_end_idxs.flatten(-2)).reshape(copy_end_idxs.shape) # B x max_in_len x max_in_len
                marginalized_at_pos_copy_logprobs = copy_span_logprobs[:, i] + generation_after_copy_logprob

                # Mask invalid copy actions
                marginalized_at_pos_copy_logprobs.masked_fill_(mask=(1 - copyable_spans[:, i]).bool(), value=float('-inf'))

                action_logprobs = torch.cat([
                    (generation_target_logprob[:, i] + (logprob_at_position[:, i+1] if i < max_out_length -1 else 0)).unsqueeze(-1),
                    marginalized_at_pos_copy_logprobs.flatten(-2)
                    ], dim=-1) # B x 1 + max_inp_len * max_inp_len

                length_mask = (i < sequence_lengths-1).float()   # -1 accounts for the <s> symbol
                logprob_at_position = logprob_at_position.clone()  # To allow for grad propagation
                logprob_at_position[:, i] = torch.logsumexp(action_logprobs, dim=-1).clamp(min=float('-inf'), max=0) * length_mask

            correct_seq_gen_logprob = logprob_at_position[:, 0]  # B
        else:
            copy_target_logprobs = copy_span_logprobs.masked_fill(mask=(1 - copyable_spans).bool(), value=float('-inf')) # B x max_out_len-1 x max_in_len x max_in_len
            copy_target_logprobs = copy_target_logprobs.flatten(2).logsumexp(dim=-1)  # B x max_out_len-1
            correct_seq_gen_logprob = torch.logsumexp(torch.stack([generation_target_logprob, copy_target_logprobs], dim=-1), dim=-1)  # B x max_out_len-1
            correct_seq_gen_logprob = correct_seq_gen_logprob.sum(dim=-1)
        if normalize:
            correct_seq_gen_logprob = correct_seq_gen_logprob / (sequence_lengths-1).float()

        if return_debug_info:
            return correct_seq_gen_logprob, {
                'generation_logprobs': generation_logprobs.cpu().numpy(),
                'copy_span_logprobs': copy_span_logprobs.cpu().numpy(),
                'vocabulary': self.target_token_encoder.vocabulary
            }
        return correct_seq_gen_logprob


    def greedy_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                      max_length: int = 50, additional_decoder_input=None) -> List[Tuple[List[str], float]]:
        vocabulary = self.target_token_encoder.vocabulary

        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        initial_state = torch.reshape(initial_state,
            (initial_state.shape[0], self.get_hyperparameter('num_layers'), self.get_hyperparameter('hidden_size'))).transpose(0, 1).contiguous() # num_layers x B x H

        current_decoder_state = initial_state  # num_layers x B x H
        batch_size = memories.shape[0]
        next_tokens = torch.tensor(
            [[vocabulary.get_id_or_unk(self.START)]] * batch_size,
            device=self.device)  # B x 1

        is_done = np.zeros(next_tokens.shape[0], dtype=np.bool)

        predicted_tokens = []  # type: List[List[str]]
        predicted_logprobs = []  # type: List[List[float]]
        remaining_copied_span = [[] for _ in range(batch_size)]  # type: List[List[str]]
        actions_taken = [[] for _ in range(batch_size)]

        for i in range(max_length):
            if np.all(is_done):
                max_length = i
                break
            # Embed next_tokens
            target_token_embeddings, _ = self.target_token_encoder.forward(
                as_packed_sequence=False,
                token_ids=next_tokens,
                lengths=None)  # B x 1 x D
            generation_logprobs, copy_span_logprobs, current_decoder_state = self.__get_output_logprobs(
                target_token_embeddings, current_decoder_state, memories, memory_lengths,
                additional_decoder_input=additional_decoder_input
                )

            generation_logprobs, token_to_generate = generation_logprobs.squeeze(1).max(dim=-1)
            generation_logprobs = generation_logprobs.cpu().numpy()
            token_to_generate = token_to_generate.squeeze(-1).cpu().numpy()  # B

            copy_span_logprobs = copy_span_logprobs.squeeze(1)  # B x max-inp-len (span-start) x max-inp-len (span-end)
            copy_target_span_logprobs, copy_span_idxs = copy_span_logprobs.flatten(start_dim=1).max(dim=-1)
            copy_target_span_logprobs = copy_target_span_logprobs.cpu().numpy()

            should_copy_logprobs = copy_target_span_logprobs
            copy_span_idxs = copy_span_idxs.cpu().numpy()
            copy_span_start = copy_span_idxs // copy_span_logprobs.shape[-1]  # B
            copy_span_end = copy_span_idxs % copy_span_logprobs.shape[-1]  # B
            assert np.all(copy_span_start <= copy_span_end)

            predicted_tokens_for_this_step = []  # type: List[str]
            predicted_logprobs_for_this_step = []  # type: List[float]
            for j in range(batch_size):
                if len(remaining_copied_span[j]) > 0:
                    # We still have a copied span, keep copying...
                    predicted_tokens_for_this_step.append(remaining_copied_span[j][0])
                    remaining_copied_span[j] = remaining_copied_span[j][1:]
                    predicted_logprobs_for_this_step.append(0)  # We have already "paid" the loss for this copy, when we decided to copy the span.
                else:
                    if should_copy_logprobs[j] >= generation_logprobs[j]:
                        # We copy
                        span_to_be_copied = memories_str_representations[j][copy_span_start[j]: copy_span_end[j]+1]
                        predicted_tokens_for_this_step.append(span_to_be_copied[0])
                        remaining_copied_span[j] = span_to_be_copied[1:]
                        predicted_logprobs_for_this_step.append(copy_target_span_logprobs[j])
                        actions_taken[j].append(f'Copy Span {span_to_be_copied}')
                    else:
                        # We generate a token
                        target_token_idx = token_to_generate[j]
                        target_token = vocabulary.get_name_for_id(target_token_idx)
                        predicted_tokens_for_this_step.append(target_token)
                        predicted_logprobs_for_this_step.append(generation_logprobs[j])
                        actions_taken[j].append(f'Generating Token {target_token}')

                if predicted_tokens_for_this_step[-1] == self.END:
                    is_done[j] = True

            predicted_logprobs.append(predicted_logprobs_for_this_step)
            predicted_tokens.append(predicted_tokens_for_this_step)
            next_tokens = torch.tensor([[vocabulary.get_id_or_unk(t)] for t in predicted_tokens_for_this_step],
                                                device=self.device)

        # Finally, convert ids to to strings
        predictions = []  # type: List[Tuple[List[List[str]], List[float]]]
        for i in range(batch_size):
            tokens = [predicted_tokens[j][i] for j in range(max_length)]
            try:
                end_idx = tokens.index(self.END)
            except ValueError:
                end_idx = max_length
            tokens = tokens[:end_idx]
            logprob = sum(float(predicted_logprobs[j][i]) for j in range(min(end_idx+1, max_length)))
            predictions.append(([tokens], [logprob]))
            # print(tokens, logprob, actions_taken[i])
        return predictions

    def beam_decode(self, memories, memory_lengths, initial_state, memories_str_representations: List[List[str]],
                    max_length: int = 150, beam_size: int = 20, max_search_size: int = 50, additional_decoder_input=None) -> List[Tuple[List[List[str]], List[float]]]:
        vocabulary = self.target_token_encoder.vocabulary
        if self.__include_summarizing_network:
            initial_state = self.__summarization_layer(initial_state)
        initial_state = torch.reshape(initial_state,
            (memories.shape[0], self.get_hyperparameter('num_layers'), self.get_hyperparameter('hidden_size'))).transpose(0, 1).contiguous() # num_layers x B x H

        current_decoder_state = initial_state  # num_layers x B x H
        batch_size = memories.shape[0]
        next_tokens = torch.tensor(
            [[vocabulary.get_id_or_unk(self.START)]] * batch_size,
            device=self.device).unsqueeze(-1)  # B x beam_size=1 x 1

        is_done = [np.zeros(1, dtype=np.bool) for _ in range(batch_size)] # B x current_beam_size

        predicted_tokens = [[[] for _ in range(beam_size)] for _ in range(batch_size)]  # type: List[List[List[str]]]
        predicted_logprobs = [[0 for _ in range(beam_size)] for _ in range(batch_size)]  # type: List[List[float]]
        remaining_copied_span = [[[] for _ in range(beam_size)] for _ in range(batch_size)]  # type: List[List[List[str]]]
        actions_taken = [[[] for _ in range(beam_size)] for _ in range(batch_size)]  # type: List[List[List[str]]]

        for i in range(max_length):
            if np.all(is_done):
                break

            current_beam_size = (next_tokens.shape[0] * next_tokens.shape[1]) // batch_size
            # Embed next_tokens
            target_token_embeddings, _ = self.target_token_encoder.forward(
                as_packed_sequence=False,
                token_ids=next_tokens.flatten(0, 1),
                lengths=None)  # B * current_beam_size x 1 x D
            generation_logprobs, copy_span_logprobs, current_decoder_state = self.__get_output_logprobs(
                target_token_embeddings, current_decoder_state,
                memories=memories.unsqueeze(1).expand(-1, current_beam_size, -1, -1).flatten(0,1),
                memory_lengths=memory_lengths.unsqueeze(1).expand(-1, current_beam_size).flatten(0, 1),
                additional_decoder_input=additional_decoder_input.unsqueeze(1).expand(-1, current_beam_size, -1).flatten(0, 1) if additional_decoder_input is not None else None
            )

            current_decoder_state = current_decoder_state.transpose(0, 1).reshape(
                batch_size, current_beam_size, self.get_hyperparameter('num_layers'),  -1)  # B x current_beam_size x num_layers x H

            generation_logprobs, token_to_generate = generation_logprobs.squeeze(1).topk(max_search_size, dim=-1)
            generation_logprobs = generation_logprobs.reshape(batch_size, current_beam_size, max_search_size).cpu().numpy()  # B x current_beam_size x max_search_size
            token_to_generate = token_to_generate.squeeze(-1).reshape(batch_size, current_beam_size, max_search_size).cpu().numpy()  # B x current_beam_size x max_search_size

            copy_span_logprobs = copy_span_logprobs.squeeze(1)  # B*current_beam_size x max-inp-len (span-start) x max-inp-len (span-end)
            num_topk = min(max_search_size, copy_span_logprobs.shape[-1] * (copy_span_logprobs.shape[-1] - 1) // 2)
            copy_target_span_logprobs, copy_span_idxs = copy_span_logprobs.flatten(start_dim=1).topk(num_topk, dim=-1)
            copy_target_span_logprobs = copy_target_span_logprobs.reshape(batch_size, current_beam_size, num_topk).cpu().numpy()  # B x current_beam_size x num_topk

            copy_span_idxs = copy_span_idxs.reshape(batch_size, current_beam_size, num_topk).cpu().numpy()
            copy_span_start = copy_span_idxs // copy_span_logprobs.shape[-1]  # B x current_beam_size x max_search_size
            copy_span_end = copy_span_idxs % copy_span_logprobs.shape[-1]  # B x current_beam_size x max_search_size
            assert np.all(copy_span_start <= copy_span_end)

            predicted_tokens_for_this_step = []
            next_decoder_state = np.zeros((batch_size, beam_size) + current_decoder_state.shape[-2:], dtype=np.float)

            for j in range(batch_size):
                beam_predicted_tokens = []
                beam_predicted_logprobs = []
                beam_remaining_copied_spans = []
                beam_is_done = []
                beam_state_idx = []
                beam_action_taken = []
                for k in range(current_beam_size):
                    if is_done[j][k]:
                        beam_predicted_tokens.append(predicted_tokens[j][k])
                        beam_remaining_copied_spans.append([])
                        beam_predicted_logprobs.append(predicted_logprobs[j][k])
                        beam_is_done.append(True)
                        beam_state_idx.append(k)
                        beam_action_taken.append(actions_taken[j][k])
                        continue

                    if len(remaining_copied_span[j][k]) > 0:
                        # We still have a copied span, keep copying... the beam of suggestions makes no sense here.
                        beam_predicted_tokens.append(predicted_tokens[j][k] + remaining_copied_span[j][k][:1])
                        beam_remaining_copied_spans.append(remaining_copied_span[j][k][1:])
                        beam_predicted_logprobs.append(predicted_logprobs[j][k])  # We have already "paid" the loss for this copy, when we decided to copy the span.
                        beam_is_done.append(remaining_copied_span[j][k][0] == self.END)
                        beam_state_idx.append(k)
                        beam_action_taken.append(actions_taken[j][k])
                    else:
                        for l in range(num_topk):
                            # Option 1: We copy
                            span_to_be_copied = memories_str_representations[j][copy_span_start[j, k, l]: copy_span_end[j, k, l] + 1]
                            assert copy_span_start[j, k, l] < copy_span_end[j, k, l] + 1
                            assert len(span_to_be_copied) > 0
                            beam_predicted_tokens.append(predicted_tokens[j][k] + span_to_be_copied[:1])
                            beam_remaining_copied_spans.append(span_to_be_copied[1:])
                            beam_predicted_logprobs.append(predicted_logprobs[j][k] + copy_target_span_logprobs[j, k, l])
                            beam_is_done.append(span_to_be_copied[0]==self.END)
                            beam_state_idx.append(k)
                            beam_action_taken.append(actions_taken[j][k] + ['Copy Span ' + str(span_to_be_copied)])

                            # Option 2: We generate a token
                            target_token_idx = token_to_generate[j, k, l]
                            target_token = vocabulary.get_name_for_id(target_token_idx)
                            # There are rare cases where an empty sequence is predicted. Explicitly add the END token in those cases.
                            beam_predicted_tokens.append(predicted_tokens[j][k] + [target_token])
                            beam_is_done.append(target_token == self.END)
                            beam_remaining_copied_spans.append([])
                            beam_predicted_logprobs.append(predicted_logprobs[j][k] + generation_logprobs[j, k, l])
                            beam_state_idx.append(k)
                            beam_action_taken.append(actions_taken[j][k] + ['Generate ' + target_token])

                # Merge identical beams
                idxs_for_next_beam = np.argsort(beam_predicted_logprobs)[::-1]  # Descending order
                to_ignore = set()
                for k in idxs_for_next_beam:
                    if k in to_ignore:
                        continue
                    for l in range(k+1, len(beam_predicted_tokens)):
                        if beam_predicted_tokens[k] == beam_predicted_tokens[l] and len(beam_remaining_copied_spans[l]) == 0 and len(beam_remaining_copied_spans[k]) == 0:
                            # l can and should be merged into k
                            to_ignore.add(l)
                            beam_predicted_logprobs[k] = min(0, np.logaddexp(beam_predicted_logprobs[k], beam_predicted_logprobs[l]))
                            if len(beam_action_taken[l]) < len(beam_action_taken[k]):
                                beam_action_taken[k] = beam_action_taken[l]

                # Now merge all the beams and pick the top beam_size elements
                idxs_for_next_beam = np.argsort(beam_predicted_logprobs)[::-1]
                idxs_for_next_beam = [idx for idx in idxs_for_next_beam if idx not in to_ignore][:beam_size]  # Remove merged idxs

                if len(idxs_for_next_beam) < beam_size:
                    # In some cases, we won't have enough elements to fill in the beam (due to merging). Fill them with dummy elements
                    beam_predicted_logprobs.append(float('-inf'))
                    idxs_for_next_beam = idxs_for_next_beam + [-1] * (beam_size - len(idxs_for_next_beam))

                predicted_tokens[j] = [beam_predicted_tokens[k] for k in idxs_for_next_beam]
                predicted_logprobs[j] = [beam_predicted_logprobs[k] for k in idxs_for_next_beam]
                actions_taken[j] = [beam_action_taken[k] for k in idxs_for_next_beam]

                remaining_copied_span[j] = [beam_remaining_copied_spans[k] for k in idxs_for_next_beam]
                is_done[j] = np.array([beam_is_done[k] for k in idxs_for_next_beam], dtype=np.bool)

                predicted_tokens_for_this_step.append([beam_predicted_tokens[k][-1] for k in idxs_for_next_beam])
                for k, idx in enumerate(idxs_for_next_beam):
                    next_decoder_state[j, k] = current_decoder_state[j, beam_state_idx[idx]].cpu().numpy()

            next_tokens = torch.tensor([[vocabulary.get_id_or_unk(t) for t in r] for r in predicted_tokens_for_this_step], device=self.device).unsqueeze(-1)
            current_decoder_state = torch.tensor(next_decoder_state, device=self.device, dtype=torch.float32).flatten(0, 1).reshape(
                (self.get_hyperparameter('num_layers'), -1, self.get_hyperparameter('hidden_size'))
            )  # num_layers x B * beam_size x H

        # Probe for visualizing decoding
        for i in range(batch_size):
            print('\nInput: ', ' '.join(memories_str_representations[i]))
            for j in range(2):
                print(f'>Pred{j+1}', ' '.join(predicted_tokens[i][j]), predicted_logprobs[i][j], actions_taken[i][j])
        return [([s[:-1] for s in predicted_tokens[i]], predicted_logprobs[i]) for i in range(batch_size)]
