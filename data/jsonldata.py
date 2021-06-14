from typing import Iterator, Dict, Union, List
from collections import Counter
import numpy as np

from dpu_utils.utils import RichPath

from data.edits import Edit, NLEdit


def parse_jsonl_edit_data(path: RichPath) -> Iterator[Edit]:
    for line in path.read_as_jsonl():
        yield Edit(
            input_sequence=line['input_sequence'],
            output_sequence=line['output_sequence'],
            provenance=line.get('provenance', ''),
            edit_type=line.get('edit_type', '')
        )

def parse_monolingual_edit_data(path: RichPath) -> Iterator[Edit]:
    for i, line in enumerate(path.read_as_jsonl()):
        yield Edit(
            input_sequence=None,
            output_sequence=line,
            provenance=f'L{i}',
            edit_type=''
        )

def make_random_edit(text: List[str], cache: List[str]) -> List[str]:
    rnd_num = np.random.rand()
    if rnd_num < 0.2:
         # no edit (20%)
        return text
    elif rnd_num < 0.4:
        # Delete random element (20%)
        num_deletions = np.random.random_integers(1, 2)
        deleted_text = list(text)
        for _ in range(num_deletions):
            rnd_pos = np.random.randint(len(deleted_text))
            deletion_size = np.random.randint(1, 4-num_deletions)
            deleted_text = deleted_text[:rnd_pos] + deleted_text[rnd_pos+deletion_size:]
        return deleted_text
    elif rnd_num < 0.6:
        # Swap two consecutive words (20%)
        swapped_text = list(text)
        rnd_pos = np.random.randint(len(swapped_text)-1)
        swapped_text[rnd_pos], swapped_text[rnd_pos+1] = swapped_text[rnd_pos], swapped_text[rnd_pos+1]
        return swapped_text
    elif rnd_num < 0.8:
        # Swap with a word in the cache (20%)
        swapped_text = list(text)
        rnd_pos = np.random.randint(len(swapped_text))
        num_swaps = np.random.randint(1, 3)
        for _ in range(num_swaps):
            rnd_pos = np.random.randint(len(swapped_text))
            swapped_text[rnd_pos] = cache[np.random.randint(len(cache))]
        return swapped_text
    else:
        # Add random word (20%)
        rnd_pos = np.random.randint(0, len(text))
        return text[:rnd_pos] + [cache[np.random.randint(len(cache))]] + text[rnd_pos:]

def parse_monolingual_synthetic_edit_data(path: RichPath) -> Iterator[Edit]:
    word_cache = []
    for i, line in enumerate(path.read_as_jsonl()):
        if len(line) < 3:
            continue
        word_cache.extend(line)
        yield Edit(
            input_sequence=make_random_edit(line, word_cache),
            output_sequence=line,
            provenance=f'L{i}',
            edit_type=''
        )

        if np.random.rand() < .01:
            # Clean pseudo-cache
            np.random.shuffle(word_cache)
            word_cache = word_cache[:2000]

def save_jsonl_edit_data(data: Iterator[Union[Edit, Dict]], path: RichPath) -> None:
    def to_dict():
        for edit in data:
            if isinstance(edit, Edit):
                yield edit._asdict()
            elif isinstance(edit, dict):
                yield edit
            else:
                raise ValueError('Unrecognized input data')

    path.save_as_compressed_file(to_dict())
