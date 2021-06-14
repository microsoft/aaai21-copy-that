from typing import Optional, Dict, Any, Tuple, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence

from mlcomponents.embeddings import SequenceEmbedder
from .sequenceencoder import SequenceEncoder


class BiGruSequenceEncoder(SequenceEncoder):
    def __init__(self, name: str, token_embedder: SequenceEmbedder,
                 hyperparameters: Optional[Dict[str, Any]]=None) -> None:
        super(BiGruSequenceEncoder, self).__init__(name, token_embedder, hyperparameters)
        self.__birnn = None  # type: Optional[nn.GRU]

    def _finalize_component_metadata_and_model(self) -> None:
        num_layers = self.get_hyperparameter('num_layers')
        hidden_size = self.get_hyperparameter('hidden_size')
        self.__birnn = nn.GRU(input_size=self.token_embedder.embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=self.get_hyperparameter('dropout_rate'))

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'hidden_size': 32,
            'num_layers': 1,
            'dropout_rate': 0.2
        }

    @property
    def summary_state_size(self) -> int:
        return 2 * self.get_hyperparameter('num_layers') * self.get_hyperparameter('hidden_size')

    @property
    def output_states_size(self) -> int:
        return 2 * self.get_hyperparameter('hidden_size')

    def forward(self, *, input_sequence_data: Dict[str, Any], return_embedded_sequence: bool=False) \
            -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        embedded_tokens, reverse_sortmap = self.token_embedder.forward(**input_sequence_data)  # type: PackedSequence

        outputs, hn = self.__birnn.forward(embedded_tokens)

        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        lengths = lengths.to(self.device)  # Oddly, even if the outputs are on the GPU, these are on the CPU.
        outputs = outputs[reverse_sortmap]
        lengths = lengths[reverse_sortmap]

        # Transform
        h_seq = hn.transpose(1, 0)  # B x num_layers * 2 x hidden_size
        h_seq = h_seq.contiguous().view(h_seq.shape[0], -1)  # B x 2 * num_layers * hidden_size
        h_seq = h_seq[reverse_sortmap]

        if return_embedded_sequence:
            embeddings, _ = pad_packed_sequence(embedded_tokens, batch_first=True)
            return outputs, lengths, h_seq, embeddings[reverse_sortmap]
        return outputs, lengths, h_seq
