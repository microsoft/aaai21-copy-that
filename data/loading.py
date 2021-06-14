from typing import Iterable, Callable

from dpu_utils.utils import RichPath

from data import fcedataloader as fcedataloader, codadataloader as codedataloader, \
    wikieditsloader as wikiatomiceditsloader, paraphraseloader
from data.edits import Edit
from data.jsonldata import parse_jsonl_edit_data, parse_monolingual_edit_data, parse_monolingual_synthetic_edit_data
from data.datautils import LazyDataIterable
from data.m2loader import parse_m2_folder


def load_data_by_type(path: RichPath, data_type: str, cleanup: bool=False, as_list: bool=True) -> Iterable[Edit]:
    def pkg(x: Callable):
        if as_list:
            return list(x())
        else:
            return LazyDataIterable(x)

    if data_type == 'fce':
        return pkg(lambda: fcedataloader.load_data_from(path))
    elif data_type == 'code':
        return pkg(lambda: codedataloader.load_data_from(path))
    elif data_type == 'codecontext':
        # Returns List[EditContext] (includes one extra field)
        return pkg(lambda: codedataloader.load_data_with_context_from(path))
    elif data_type == 'fixer':
        return pkg(lambda: codedataloader.load_fixer_data(path))
    elif data_type == 'wikiatomicedits':
        return pkg(lambda: wikiatomiceditsloader.load_data_from(path, 4000000, remove_identical=cleanup))
    elif data_type == 'wikiedits':
        return pkg(lambda: wikiatomiceditsloader.load_data_from(path, has_phrase=False, remove_identical=cleanup))
    elif data_type == 'paraphrase':
        return pkg(lambda: paraphraseloader.load_data_from(path, remove_identical=cleanup))
    elif data_type == 'jsonl':
        return pkg(lambda: parse_jsonl_edit_data(path))
    elif data_type == 'm2':
        return pkg(lambda: parse_m2_folder(path))
    elif data_type == 'monolingual':
        return pkg(lambda : parse_monolingual_edit_data(path))
    elif data_type == 'monolingual-synth-edits':
        return pkg(lambda : parse_monolingual_synthetic_edit_data(path))
    else:
        raise ValueError('Unrecognized data type %s' % data_type)
