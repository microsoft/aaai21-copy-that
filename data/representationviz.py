import json
import os
from typing import List, Callable, TypeVar, Generic, Optional

import numpy as np
from annoy import AnnoyIndex
from sklearn.manifold import TSNE

T = TypeVar('T')

class RepresentationsVisualizer(Generic[T]):
    def __init__(self, labeler: Callable[[T], str], colorer: Callable[[T], str]=None, distance_metric: str='euclidean'):
        self.__labeler = labeler
        self.__colorer = colorer
        self.__distance_metric = distance_metric

    def print_nearest_neighbors(self, datapoints: List[T], representations: np.ndarray, num_neighbors: int=2,
                                num_items: Optional[int]=None, datapoint_to_string: Optional[Callable[[T], str]]=None):
        assert len(datapoints) == representations.shape[0], 'Number of datapoints and representations do not match.'

        for i, nns, distances in self.compute_nearest_neighbors(datapoints, representations, num_neighbors, num_items):
            print('-------------------------------------------------------')
            print(f'Target: {datapoints[i] if datapoint_to_string is None else datapoint_to_string(datapoints[i])}')
            for j, (nn, dist) in enumerate(zip(nns, distances)):
                print(f'Neighbor {j+1} (distance={dist:.2f}) {datapoints[nn] if datapoint_to_string is None else datapoint_to_string(datapoints[nn])}')

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
                'nns': [dict(diff=datapoint_to_html(datapoints[nn]),dist='%.3f' % dist) for nn, dist in zip(nns, distances)]
            })
            if i > num_items_to_show:
                break

        with open(os.path.join(os.path.dirname(__file__), 'diffviz.mustache')) as f:
            diff_template = pystache.parse(f.read())

        renderer = pystache.Renderer()
        html = renderer.render(diff_template, dict(samples=nns_viz_data))
        print('Writing output at %s' % outfile)
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(html)

    @staticmethod
    def square_to_condensed(i, j, n):
        assert i != j, "no diagonal elements in condensed matrix"
        if i < j:
            i, j = j, i
        return int(n * j - j * (j + 1) / 2 + i - 1 - j)

    def compute_nearest_neighbors(self, datapoints, representations, num_neighbors: int, num_items: Optional[int]=None, distance_threshold: float=.6):
        index = AnnoyIndex(representations.shape[1], metric='angular')
        for i in range(len(datapoints)):
            index.add_item(i, representations[i])
        index.build(50)  # TODO: Fine tune this hyper
        print('Nearest neighbor index built.')

        num_items_shown = 0
        for i, data in enumerate(datapoints):
            if num_items is not None and i > num_items:
                break
            nns, distances = index.get_nns_by_item(i, num_neighbors+1, include_distances=True)
            if nns[0] == i:
                distance_of_first = distances[1]
            else:
                distance_of_first = distances[0]

            if distance_of_first > distance_threshold:
                continue

            num_items_shown += 1
            if nns[0] == i:
                yield i, nns[1:], distances[1:]
            else:
                yield i, nns, distances

    def save_tsne_as_json(self, datapoints, representations, save_file: str):
        emb_2d = TSNE(n_components=2, verbose=1, metric=self.__distance_metric).fit_transform(representations)

        out_dict = []
        for i in range(len(datapoints)):
            out_dict.append(
                {
                    'xy': [float(emb_2d[i][0]), float(emb_2d[i][1])],
                    'before': datapoints[i]['input_sequence'],
                    'after': datapoints[i]['output_sequence'],
                    'label': self.__labeler(datapoints[i]),
                    'color': self.__colorer(datapoints[i]) if self.__colorer is not None else ''
                }
            )

        with open(save_file, 'w') as f:
            json.dump(out_dict, f)

    def plot_tsne(self, datapoints, representations, save_file: Optional[str]=None):
        emb_2d = TSNE(n_components=2, verbose=1, metric=self.__distance_metric).fit_transform(representations)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,20))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1])
        for i in range(len(datapoints)):
            plt.annotate(s=self.__labeler(datapoints[i]), xy=(emb_2d[i, 0], emb_2d[i, 1]))

        if save_file is None:
            plt.show()
        else:
            plt.savefig(save_file)
