import sys

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from dpu_utils.utils import RichPath

from data.edits import Edit
from dpu_utils.ptutils import BaseComponent

'''
# Copy Span Visualization
'''
model_path = sys.argv[1]

@st.cache
def get_model(filename):
    path = RichPath.create(filename)
    model = BaseComponent.restore_model(path, device='cpu')
    model.eval()

    return model

st.markdown(f'> Using model from {model_path}')
before_tokens = st.text_area('Input (space) tokenized before version.').strip().split()
after_tokens = st.text_area('Input (space) tokenized after version.').strip().split()

'''
#### Input Data
'''
edit = Edit(input_sequence=before_tokens, output_sequence=after_tokens, provenance='', edit_type='')
st.write(edit)

model = get_model(model_path)

tensorized_data = [model.load_data_from_sample(edit)]
mb_data, is_full, num_elements = model.create_minibatch(tensorized_data, max_num_items=10)
assert num_elements == 1

ground_input_sequence = [edit.input_sequence]
predicted_outputs = model.beam_decode(input_sequences=mb_data['input_sequences'],
                                                        ground_input_sequences=ground_input_sequence)[0]

ll, debug_info = model.compute_likelihood(**mb_data, return_debug_info=True)
ll = ll.cpu().numpy()
st.markdown(f' > Likelihood of target edit {ll[0]:.2f}')

copy_span_logprobs = debug_info['copy_span_logprobs'][0]
gen_logprobs = debug_info['generation_logprobs'][0]
vocabulary = debug_info['vocabulary']
before_tokens = ['<s>'] + before_tokens + ['</s>']
after_tokens = after_tokens + ['</s>']

for i in range(copy_span_logprobs.shape[0]):
    st.markdown(f'### At position {i}: "{after_tokens[i]}"')
    st.markdown(f'Current context `{["<s>"] + after_tokens[:i]}`')
    plt.figure(figsize=[1, 1])
    current_copy_span_probs = np.exp(copy_span_logprobs[i])
    plt.matshow(current_copy_span_probs, cmap='Greys')
    plt.xticks(range(copy_span_logprobs.shape[1]), before_tokens, fontsize=8, rotation=90)
    plt.xlabel('Start Span Pos')
    plt.yticks(range(copy_span_logprobs.shape[2]), before_tokens, fontsize=8, rotation=0)
    plt.ylabel('End Span Pos')
    plt.colorbar()
    st.pyplot()

    max_idx = np.argmax(current_copy_span_probs)
    from_idx, to_idx = max_idx // current_copy_span_probs.shape[1], max_idx % current_copy_span_probs.shape[1],
    st.markdown(f'* Best copy suggestion: `Copy({from_idx}:{to_idx+1})` with prob {np.max(current_copy_span_probs)*100:.1f}%, _i.e._ `Copy({before_tokens[from_idx: to_idx+1]})`.')
    st.markdown(f'* Best generation suggestion: `Gen("{vocabulary.get_name_for_id(np.argmax(gen_logprobs[i]))}")` with prob {np.exp(np.max(gen_logprobs[i]))*100:.1f}%')

'''### Beam decoding results '''
for i, (prediction, logprob) in enumerate(zip(predicted_outputs[0], predicted_outputs[1])):
    if i > 2:
        break
    st.markdown(f'* {" ".join(prediction)} ({np.exp(logprob)*100:.1f}%)')
