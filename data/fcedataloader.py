import logging
from typing import Iterator, List, Tuple, NamedTuple

from dpu_utils.utils import RichPath

from data.edits import Edit


def load_data_from(file: RichPath) -> Iterator[Edit]:
    num_excluded_samples = 0
    with open(file.to_local_path().path) as f:
        for i, row in enumerate(f):
            edit_start_idx, edit_end_idx, source_words, target_words, error_type, sentence = row.split('\t')
            edit_start_idx, edit_end_idx = int(edit_start_idx), int(edit_end_idx)
            sentence = sentence.lower().split()
            source_words = source_words.lower().split()
            target_words = target_words.lower().split()
            assert sentence[edit_start_idx:edit_end_idx] == source_words
            output_sequence = sentence[:edit_start_idx] + target_words + sentence[edit_end_idx:]
            if sentence == output_sequence:
                num_excluded_samples += 1
                continue
            if len(sentence) < 2 or len(output_sequence) < 2:
                num_excluded_samples += 1
                continue
            yield Edit(
                input_sequence=sentence,
                output_sequence=output_sequence,
                edit_type=error_type,
                provenance=f'row{i}'
            )
    logging.warning('Removed %s samples because before/after sentence was identical or too small.', num_excluded_samples)