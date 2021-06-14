# Copy that! Editing Sequences by Copying Spans

### Prerequisites
Python 3.6 or higher is needed with PyTorch 1.2 or higher. The `dpu-utils` package is also necessary:
```shell
> pip3 install dpu-utils
```

### How to run
* First convert the data to the appropriate format. `data/loading.py` shows a few of the possible methods. The [`JSONL`](http://jsonlines.org/) style is probably the easiest one. This format is a `.jsonl.gz` where each line has the format:
    ```json
    {
        "input_sequence": ["list", "of", "input", "tokens", ...],
        "output_sequence": ["list", "of", "output", "tokens", ...]
    }
    ```
    Optionally, you can have a `"provenance"` field and an `"edit_type"` field.

    To define the type of data used in training/testing, use the `--data-type` option in the command line.

* Then run training
    ```shell
    > python3 model/train.py --data-type=jsonl path/to/train/data /path/to/validation/data <modelname> ./model-save/filename.pkl.gz
    ```
    There are multiple possible `modelname`, but commonly you'd like to use the `baseseq2seq` and `basecopyspan`.

    > To run you the code need Python3 and PyTorch. Make sure that your `PYTHONPATH` environment variable points to the root folder of this repository.

* Output parallel predictions.
    ```shell
    > python3 model/outputparallelpredictions.py --data-type=jsonl ./model-save/filename.pkl.gz path/to/test/data path/for/output/txt/file_prefix
    ```
    This will output the before/after predictions in separate files in two files ending in `before.txt` and `after.txt`.

### Available Models
Models are defined in `editrepresentationmodels.py` and their constructor is invoked in `train.py`.

Editing Models:
* `basecopyseq2seq` A GRU-based sequence2sequence model with attention and (simple) copying.
* `basecopyspan` A GRU-based sequence2sequence model with span-copying.

Edit Representation Models:

Edit representation models follow the structure of the work of [Yin et al.](https://arxiv.org/abs/1810.13337).
* `copyseq2seq` A GRU-based edit representation model with attention and (simple) copying.
* `copyspan` A GRU-based edit representation model with span-copying.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
