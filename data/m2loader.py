from typing import Iterator, List, Tuple

from dpu_utils.utils import RichPath

from data.edits import Edit

def apply_edits(original_sentence: List[str], edits: List[Tuple[int, int, List[str]]]) -> List[str]:
    edited_sentence = []
    last_edit_idx = 0

    for from_idx, to_idx, edit in edits:
        edited_sentence += original_sentence[last_edit_idx:from_idx] + edit
        last_edit_idx = to_idx
    edited_sentence += original_sentence[last_edit_idx:]
    return edited_sentence

def parse_m2_file(m2_file: RichPath) -> Iterator[Edit]:
    original_sentence = None
    edits = []
    provenance = None
    annotator_id = None
    for i, line in enumerate(m2_file.read_as_text().splitlines()):
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('S'):
            if original_sentence is not None and len(edits) > 0:
                edited_sentence = apply_edits(original_sentence, edits)
                yield Edit(input_sequence=original_sentence, output_sequence=edited_sentence,
                            provenance=provenance, edit_type='')
            original_sentence = line.split(' ')[1:]  # Remove " S"
            edits = []
            provenance = m2_file.path + 'L' + str(i)
            annotator_id = None
        elif line.startswith('A '):
            range, edit_type, replacement, _, _, next_annotator_id = line[2:].split('|||')
            if edit_type == 'noop':
                yield Edit(input_sequence=original_sentence, output_sequence=original_sentence,
                               provenance=provenance, edit_type='')
                continue

            if annotator_id != next_annotator_id and annotator_id is not None:
                edited_sentence = apply_edits(original_sentence, edits)
                yield Edit(input_sequence=original_sentence, output_sequence=edited_sentence,
                               provenance=provenance, edit_type='')
                edits = []

            annotator_id = next_annotator_id

            start_idx, end_idx = range.split()
            start_idx, end_idx = int(start_idx), int(end_idx)
            replacement = replacement.split()
            if start_idx == end_idx and len(replacement) == 0:
                continue
            edits.append((start_idx, end_idx, replacement))

    # Last edit.
    if original_sentence is not None:
        edited_sentence = apply_edits(original_sentence, edits)
        yield Edit(input_sequence=original_sentence, output_sequence=edited_sentence,
                       provenance=provenance, edit_type='')


def parse_m2_folder(folder: RichPath)  -> Iterator[Edit]:
    for m2_file in folder.iterate_filtered_files_in_dir('*.m2'):
        yield from parse_m2_file(m2_file)
