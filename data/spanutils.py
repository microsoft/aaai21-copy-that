from typing import List

import numpy as np

def get_copyable_spans(input: List[str], output: List[str]) -> np.ndarray:
    """
    Return a 3D tensor copy_mask[k, i, j] that for a given location k shows the all the possible
       spans that can be copied.

       All valid start locations can be obtained at point k by diag(copy_masks[k])

       All the possible end positions at position k, given a starting span i are given by copy_mask[k, i]
    """
    copy_masks = np.zeros((len(output), len(input), len(input)), dtype=np.bool)  # out-len x start_pos x end_pos
    for k in range(len(output)-1, -1, -1):
        for i in range(len(input)):
            if input[i] == output[k]:
                if k + 1 < len(output) and i + 1 < len(input):
                    # Everything valid end point at k+1 is also valid here
                    copy_masks[k, i] = copy_masks[k+1, i+1]
                copy_masks[k, i, i] = True

    return copy_masks
