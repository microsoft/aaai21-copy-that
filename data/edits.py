import difflib
from enum import Enum
from typing import NamedTuple, TypeVar, Optional, List, Dict

import enum

Edit = NamedTuple('Edit', [
    ('input_sequence', List[str]),
    ('output_sequence', List[str]),
    ('provenance', str),
    ('edit_type', List[str])
])

NLEdit = NamedTuple('NLEdit', [
    ('input_sequence', List[str]),
    ('output_sequence', List[str]),
    ('nl_sequence', List[str]),
    ('provenance', str),
    ('edit_type', List[str])
])

EditContext = NamedTuple('EditContext', [
    ('input_sequence', List[str]),
    ('output_sequence', List[str]),
    ('context_sequence', List[str]),
    ('provenance', str),
    ('edit_type', List[str])
])

# TODO: Make sure this doesn't get cut by BPE
CONTEXT_SEPERATOR = '%CONTEXT_SEP%'

@enum.unique
class ChangeType(Enum):
    EQUAL = 0
    INSERT = 1
    REPLACE = 2
    DELETE = 3


T = TypeVar('T')

AlignedDiffRepresentation = NamedTuple('AlignedDiffRepresentation', [
    ('change_type', List[ChangeType]),
    ('before_tokens', List[Optional[T]]),
    ('after_tokens', List[Optional[T]])
])

def sequence_diff(before: List[T], after: List[T]) -> AlignedDiffRepresentation:
    """
    Return a linearized sequence diff as explained in Yin et al. 2019
    """
    matcher = difflib.SequenceMatcher()

    before_tokens = []  # type: List[Optional[T]]
    after_tokens = []  # type: List[Optional[T]]
    change_types = []  # type: List[ChangeType]
    matcher.set_seqs(before, after)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            change_types.extend([ChangeType.EQUAL] * (i2 - i1))
            before_tokens.extend(before[i1:i2])
            after_tokens.extend(after[j1:j2])
        elif tag == 'delete':
            change_types.extend([ChangeType.DELETE] * (i2 - i1))
            before_tokens.extend(before[i1:i2])
            after_tokens.extend([None] * (i2 - i1))
        elif tag == 'insert':
            change_types.extend([ChangeType.INSERT] * (j2 - j1))
            before_tokens.extend([None] * (j2 - j1))
            after_tokens.extend(after[j1:j2])
        elif tag == 'replace':
            largest_span_size = max(i2-i1, j2-j1)
            change_types.extend([ChangeType.REPLACE] * largest_span_size)
            before_tokens.extend(before[i1:i2] + [None] * (largest_span_size - (i2-i1)))
            after_tokens.extend(after[j1:j2] + [None] * (largest_span_size - (j2 - j1)))
        else:
            raise Exception('Unrecognized opcode %s' % tag)

    assert len(change_types) == len(before_tokens) == len(after_tokens)
    return AlignedDiffRepresentation(change_types, before_tokens, after_tokens)
