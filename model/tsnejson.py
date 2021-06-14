#!/usr/bin/env python
"""
Usage:
    tsnejson.py [options] MODEL_FILENAME TEST_DATA OUT_PATH

Options:
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --data-type=<type>         The type of data to be used. Possible options fce, code, wikiatomicedits. [default: fce]
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import logging
from typing import List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

from data.loading import load_data_by_type
from data.representationviz import RepresentationsVisualizer
from dpu_utils.ptutils import BaseComponent


def test(model: BaseComponent, test_data: List[Dict[str, Any]], out_file_path: str):
    model.eval()
    all_data = [model.load_data_from_sample(d) for d in test_data]
    data_iter = iter(all_data)
    representations = []
    is_full = True

    start_idx = 0
    while is_full:
        mb_data, is_full, num_elements = model.create_minibatch(data_iter, max_num_items=200)
        if num_elements > 0:
            representations.append(model.edit_encoder.get_summary(input_sequence_data=mb_data['aligned_edits']))
            start_idx += num_elements
        if not is_full:
            break

    all_labels = set(t.get('edit_type', '?').split('+')[0] for t in test_data)
    colormap = plt.get_cmap('Paired')
    label_to_color = {}
    for i, label in enumerate(all_labels):
        label_to_color[label] = colormap(int(float(i) / len(all_labels) * colormap.N))


    representations = np.array(representations)
    viz = RepresentationsVisualizer(labeler=lambda d: d.get('edit_type', '?').split('+')[0],
                                    colorer=lambda d: label_to_color[d.get('edit_type', '?').split('+')[0]])
    viz.save_tsne_as_json(test_data, representations, save_file=out_file_path)

def run(arguments):
    azure_info_path = arguments.get('--azure-info', None)

    model_path = RichPath.create(arguments['MODEL_FILENAME'], azure_info_path)
    model = BaseComponent.restore_model(model_path)

    test_data_path = RichPath.create(arguments['TEST_DATA'], azure_info_path)
    test_data = load_data_by_type(test_data_path, arguments['--data-type'])

    test(model, test_data, arguments['OUT_PATH'])

if __name__ == '__main__':
    args = docopt(__doc__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    run_and_debug(lambda: run(args), args.get('--debug', False))
