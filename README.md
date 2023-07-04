# lm-output-dataset
> Dataset of various language model outputs from different datasets

## Description
This repository contains a dataset of various language model outputs from different datasets. This includes both open-source and proprietary models.
The dataset is available at 🤗 [Hugging Face](https://huggingface.co/datasets/andstor/output).


## Tags
Table of available tags and their meaning

| Tag name | Description | temperature | top-p | top-k |
|----------|-------------|:-----------:|:----:|:------:|
| greedy   | Greedy decoding | 0 | 0 | 0 |
| random   | Random sampling | 1 | 1 | 0 |

## Requirements

### Dependencies
Install the Python dependencies defined in the requirements.txt.
```bash
pip install -r requirements.txt
```

### Accelerate
Setup accelerate:
```bash
accelerate config
```

### OpenAI API access
To use the OpenAI API, you need to set the `OPENAI_API_KEY` environment variable to your API key.

## Generation with Hugging Face models
The `generate.py` script will generate samples from a specified dataset with a Hugging Face model.

### Usage

```bash
usage: generate.py [-h] [--dataset_name DATASET_NAME] [--dataset_config_name DATASET_CONFIG_NAME] [--dataset_split DATASET_SPLIT] [--text_column_name TEXT_COLUMN_NAME]
                   [--reference_column_name REFERENCE_COLUMN_NAME] [--model_name_or_path MODEL_NAME_OR_PATH] [--config_name CONFIG_NAME] [--generation_config_file GENERATION_CONFIG_FILE]
                   [--tokenizer_name TOKENIZER_NAME] [--use_slow_tokenizer] [--per_device_batch_size PER_DEVICE_BATCH_SIZE] [--output_dir OUTPUT_DIR] [--seed SEED]
                   [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS] [--overwrite_cache] [--tag TAG] [--max_new_tokens MAX_NEW_TOKENS] [--max_window_size MAX_WINDOW_SIZE] [--subsamples SUBSAMPLES]
                   [--id_column_name ID_COLUMN_NAME] [--keep_columns KEEP_COLUMNS]

Do inference with a transformer model on a causal language modeling task

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library).
  --dataset_config_name DATASET_CONFIG_NAME
                        The configuration name of the dataset to use (via the datasets library).
  --dataset_split DATASET_SPLIT
                        The name of the split to use.
  --text_column_name TEXT_COLUMN_NAME
                        The column name of the dataset to use.
  --reference_column_name REFERENCE_COLUMN_NAME
                        The column name of the dataset to use as reference. If not provided, the cutoff text_column_name will be used.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models.
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the same as model_name
  --generation_config_file GENERATION_CONFIG_FILE
                        Generation config path if not the same as model_name
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not the same as model_name
  --use_slow_tokenizer  If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).
  --per_device_batch_size PER_DEVICE_BATCH_SIZE
                        Batch size (per device) for the dataloader.
  --output_dir OUTPUT_DIR
                        Where to store the final model.
  --seed SEED           A seed for reproducible training.
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                        The number of processes to use for the preprocessing.
  --overwrite_cache     Overwrite the cached dataset
  --tag TAG             The tag to use for this generation run.
  --max_new_tokens MAX_NEW_TOKENS
                        The maximum number of new tokens to generate.
  --max_window_size MAX_WINDOW_SIZE
                        The maximum number of tokens in the input.
  --subsamples SUBSAMPLES
                        The number of subsamples to use from each data example. Randomly selected. None means use all.
  --id_column_name ID_COLUMN_NAME
                        The column name of the dataset to use as id. If not provided, the index will be used.
  --keep_columns KEEP_COLUMNS
                        The column names of the dataset to keep separate by commas. If not provided, all columns will be removed.
```

### Complete generation
Complete generation is done by providing both an input data column and a reference data column. This will make the model use the whole prompt as input. By setting `--max_new_tokens` to `auto`, all the unused embedding space is used to generate as much as possible. The maximum number of tokens in the input can be truncated (from the left) by setting `--max_window_size`, thus allowing for a longer output (`--max_new_tokens`).

#### Example
The following example will generate samples from the test split of the [Humaneval](https://huggingface.co/datasets/THUDM/humaneval-x) dataset using the greedy decoding strategy. The output will be saved to the `output` directory.

```bash
accelerate launch generate.py \
--dataset_name THUDM/humaneval-x \
--dataset_config_name python \
--dataset_split test \
--text_column_name prompt \
--reference_column_name canonical_solution \
--model_name_or_path EleutherAI/gpt-j-6B \
--generation_config_file generation_config.json \
--per_device_batch_size 1 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 10 \
--tag greedy \
--max_new_tokens auto
```

### Strided generation
Stridden generation is done by only providing an input data column. This will be split into parts where each part is generated in a "sliding window" approach. Each window serves as the reference for the preceding window. The window stride (read size) is determined by the `--max_window_size` argument. The number of tokens to be generated is controlled by `--max_new_tokens`. The number of subsamples to use from each data example can be controlled by `--subsamples`. If `--subsamples` is set to `None`, all subsamples will be used.

#### Filtering
The dataset is filtered by the following criteria:
- The input is at least max_new_tokens + max_window_size long

#### Window splitting
Given an input, it is first truncated by max_new_tokens. The result is then split into window sizes of up to max_window_size. The minimum size of each window is l / math.ceil(l / max_size), where l is len(imput)-max_new_tokens. Each window is generated independently. The first window has an index of 0, the second has an index of 1, etc.

#### Example
The following example will generate samples from the test split of the [The Pile](https://pile.eleuther.ai/) dataset using the greedy decoding strategy. The input will be truncated to 512 tokens and the maximum number of new tokens will be 512. The output will be saved to the `output` directory.


```bash
accelerate launch generate.py \
--dataset_name andstor/the_pile_github \
--dataset_config_name java \
--dataset_split test \
--text_column_name text \
--model_name_or_path EleutherAI/gpt-j-6B \
--generation_config_file generation_config.json \
--per_device_batch_size 1 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 20 \
--tag greedy \
--max_window_size 512 \
--max_new_tokens 512
```

The `generation_config.json` file contains the following configuration:
```json
{
    "do_sample": false,
    "max_new_tokens": 256,
    "bos_token_id": 50256,
    "eos_token_id": 50256
}
```


## Generation with third-party models (API)
Several third-party models such as ChatGPT have been used for generating data. Scripts for generating data with these models can be found in the `notebooks` directory. Note that access to most of these requires a paid subscription. Furthermore, most are closed-source models and might not be reproducible. 


## License

Copyright © [André Storhaug](https://github.com/andstor)

This repository is licensed under the [MIT License](https://github.com/andstor/verified-smart-contracts/blob/main/LICENSE).
