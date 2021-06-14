from typing import Iterator

from dpu_utils.utils import RichPath

from data.edits import Edit, EditContext, CONTEXT_SEPERATOR


def load_data_from(file: RichPath) -> Iterator[Edit]:
    data = file.read_by_file_suffix()
    for line in data:
        yield Edit(
            input_sequence=line['PrevCodeChunkTokens'],
            output_sequence=line['UpdatedCodeChunkTokens'],
            provenance=line['Id'],
            edit_type=''
        )

def load_fixer_data(file: RichPath) -> Iterator[Edit]:
    for line in file.read_by_file_suffix():
        yield Edit(
            input_sequence= line['PrevCodeChunkTokens'],
            output_sequence= line['UpdatedCodeChunkTokens'],
            provenance=line['Id'],
            edit_type=line['Id'].split('_')[0]
        )

def load_data_with_context_from(file: RichPath) -> Iterator[EditContext]:
    data = file.read_by_file_suffix()
    for line in data:
        try:
            if 'PrecedingContextTokens' in line:
                preceding_context_tokens = line['PrecedingContextTokens']
                succeeding_context_tokens = line['SucceedingContextTokens']
            else:
                preceding_context_tokens = line['PrecedingContext']
                succeeding_context_tokens = line['SucceedingContext']
            yield EditContext(
                input_sequence=line['PrevCodeChunkTokens'],
                output_sequence=line['UpdatedCodeChunkTokens'],
                context_sequence=preceding_context_tokens + [CONTEXT_SEPERATOR] + succeeding_context_tokens,
                provenance=line['Id'],
                edit_type=''
            )
        except:
            pass
