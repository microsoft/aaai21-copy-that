import json
import os
from typing import List, Callable, TypeVar, Generic, Optional

import numpy as np
from annoy import AnnoyIndex
from sklearn.manifold import TSNE
from data.representationviz import RepresentationsVisualizer

T = TypeVar('T')

class NLRepresentationsVisualizer(RepresentationsVisualizer):
    def __init__(self, labeler: Callable[[T], str]=None, colorer: Callable[[T], str]=None, distance_metric: str='euclidean'):
        super(NLRepresentationsVisualizer, self).__init__(labeler, colorer, distance_metric)

    def nearest_neighbors_to_html(self, datapoints: List[T], representations: np.ndarray,
                                  datapoint_to_html: Callable[[T], str], outfile: str,
                                  num_neighbors: int=2, num_items: Optional[int]=None,
                                  num_items_to_show: int=10000):
        import pystache
        assert len(datapoints) == representations.shape[0], 'Number of datapoints and representations do not match.'

        nns_viz_data = []
        for i, nns, distances in self.compute_nearest_neighbors(datapoints, representations, num_neighbors, num_items):
            nns_viz_data.append({
                'num': i,
                'diff': datapoint_to_html(datapoints[i]),
                'nns': [dict(diff=datapoint_to_html(datapoints[nn]),dist=dist,
                    nl=' '.join(datapoints[nn].nl_sequence),
                    link=datapoints[nn].provenance) for nn, dist in zip(nns, distances)],
                'nl': ' '.join(datapoints[i].nl_sequence),
                'link': datapoints[i].provenance
            })
            if i > num_items_to_show:
                break

        with open(os.path.join(os.path.dirname(__file__), 'nldiffviz.mustache')) as f:
            diff_template = pystache.parse(f.read())

        renderer = pystache.Renderer()
        html = renderer.render(diff_template, dict(samples=nns_viz_data))
        print('Writing output at %s' % outfile)
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(html)