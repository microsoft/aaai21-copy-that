from typing import Optional, Dict, Any

import torch
from torch import nn

from dpu_utils.ptutils import BaseComponent


class LuongAttention(BaseComponent):
    """
    A Luong-style attention that also includes the inner product of targets-lookup
    """
    def __init__(self, name: str, hyperparameters: Optional[Dict[str, Any]]=None) -> None:
        super(LuongAttention, self).__init__(name, hyperparameters)

        self.__Wcombine = None  # type: Optional[nn.Parameter]
        self.__Wscore = None  # type: Optional[nn.Parameter]
        self.__Wpredict = None  # type: Optional[nn.Parameter]

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'memories_hidden_dimension': 64,
            'lookup_hidden_dimension': 64,
            'output_size': 64
        }

    def _load_metadata_from_sample(self, data_to_load: Any) -> None:
        pass # Nothing here

    def load_data_from_sample(self, data_to_load: Any) -> Optional[Any]:
        pass  # Nothing here

    def initialize_minibatch(self) -> Dict[str, Any]:
        pass  # Nothing here

    def extend_minibatch_by_sample(self, datapoint: Any, accumulated_minibatch_data: Dict[str, Any]) -> bool:
        pass  # Nothing here

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        pass  # Nothing here

    def _finalize_component_metadata_and_model(self) -> None:
        self.__Whd = nn.Parameter(torch.randn(self.get_hyperparameter('memories_hidden_dimension'), self.get_hyperparameter('lookup_hidden_dimension'),
                                                   dtype=torch.float, requires_grad=True))
        self.__Wout = nn.Linear(self.get_hyperparameter('memories_hidden_dimension') + self.get_hyperparameter('lookup_hidden_dimension'),
                                self.get_hyperparameter('output_size'),
                                bias=False)

    def forward(self, *, memories: torch.Tensor, memories_length: torch.Tensor,
                lookup_vectors: torch.Tensor) -> torch.Tensor:
        return self.forward_with_attention_vec(memories=memories, memories_length=memories_length, lookup_vectors=lookup_vectors)[0]

    def forward_with_attention_vec(self, *, memories: torch.Tensor, memories_length: torch.Tensor, lookup_vectors: torch.Tensor) -> torch.Tensor:
        # memories: B x max-inp-len x H
        # memories_length: B
        # look_up_vectors: B x max-out-len x D
        attention = self.get_attention_vector(lookup_vectors, memories, memories_length)  # B x max-out-len x max-inp-len

        contexts = torch.einsum('blq,bqh->blh', attention, memories)  # B x max-out-len x H
        hc = torch.cat([contexts, lookup_vectors], dim=-1)  # B x max-out-len x H
        return torch.tanh(self.__Wout(hc)), attention

    def get_attention_vector(self, lookup_vectors, memories, memories_length):
        # memories: B x max-inp-len x H
        # memories_length: B
        # look_up_vectors: B x max-out-len x D
        # Output: B x max-out-len x max-inp-len
        memories_in_d = torch.einsum('blh,hd->bld', memories, self.__Whd)  # B x max-inp-len x D
        logits = torch.einsum('bld,bqd->bql', memories_in_d, lookup_vectors)  # B x max-out-len x max-inp-len

        mask = (torch.arange(memories.shape[1], device=self.device).view(1, -1) >= memories_length.view(-1, 1)).unsqueeze(1) # B x 1 x max-inp-len
        logits.masked_fill_(mask, float('-inf'))
        attention = nn.functional.softmax(logits, dim=-1)  # B x max-len
        return attention
