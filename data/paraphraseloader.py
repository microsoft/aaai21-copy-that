import logging
from typing import Optional, Iterator, List

from dpu_utils.utils import RichPath

from data.edits import Edit

def clean_up_sentence(tokens: List[str]) -> List[str]:
    # Remove empty spaces
    return [t.strip() for t in tokens if len(t.strip()) > 0]

def load_data_from(file: RichPath, max_size_to_load: Optional[int]=None, remove_identical: bool=True) -> Iterator[Edit]:
    num_removed = 0
    with open(file.to_local_path().path, 'r') as f:
        for i, row in enumerate(f):
            if i == 0: continue  # ignore header
            if max_size_to_load is not None and i >= max_size_to_load: break
            is_para, orig_id, target_id, original_sentence, target_sentence = row.split('\t')

            input_sentence = original_sentence.strip().lower().split()
            output_sentence = target_sentence.strip().lower().split()

            input_sentence = clean_up_sentence(input_sentence)
            output_sentence = clean_up_sentence(output_sentence)

            if remove_identical and input_sentence == output_sentence:
                num_removed += 1
                continue

            yield Edit(
                input_sequence=input_sentence,
                output_sequence=output_sentence,
                edit_type=f'{is_para}',
                provenance=f'{i}:{orig_id}->{target_id}'
            )
    if num_removed > 0:
        logging.info('Removed %s samples because they differed only in whitespace.', num_removed)
