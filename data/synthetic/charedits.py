from typing import List

import numpy as np

from data.edits import Edit

all_chars = [chr(65+i) for i in range(26)] + [chr(97+i) for i in range(26)]

def create_random_sequences(min_size: int, max_size: int, num_sequences_per_size: int):
    for seq_size in range(min_size, max_size):
        all_input_seqs = set()
        while len(all_input_seqs) < num_sequences_per_size:
            sample = np.random.choice(all_chars, size=seq_size, replace=False)
            all_input_seqs.add(tuple(sample))
        yield from all_input_seqs


##### Operations
def add_char(input_sequence: List[str]) -> List[str]:
    pos = np.random.randint(len(input_sequence)+1)
    char = np.random.choice(all_chars)
    return input_sequence[:pos] + [char] + input_sequence[pos:]

def remove_char(input_sequence: List[str]) -> List[str]:
    pos = np.random.randint(len(input_sequence))
    return input_sequence[:pos] + input_sequence[pos+1:]

def swap_char(input_sequence: List[str]) -> List[str]:
    pos = np.random.randint(len(input_sequence))
    char = np.random.choice(all_chars)
    return input_sequence[:pos] + [char] + input_sequence[pos+1:]

edit_choices = [add_char, remove_char, swap_char]

def apply_random(input_sequence: List[str]) -> Edit:
    edit_op = np.random.choice(edit_choices)
    return Edit(input_sequence=input_sequence, output_sequence=edit_op(input_sequence),
                edit_type=edit_op.__name__, provenance='')

def get_dataset():
    input_sequences = (list(s) for s in create_random_sequences(3, 10, 5000))
    all_edits = [apply_random(s) for s in input_sequences]
    return all_edits
