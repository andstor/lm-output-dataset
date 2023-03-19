# lm-output-dataset
> Dataset of various language model outputs from different datasets


The dataset is available at 🤗 [Hugging Face](https://huggingface.co/datasets/andstor/output).


## Dataset creation

Setup accelerate:
```bash
accelerate config
```

Run the script:
```bash
accelerate launch generate.py \
--dataset_name andstor/the_pile_github \
--dataset_config_name python \
--dataset_split train \
--text_column_name text \
--model_name_or_path gpt2-xl \
--generation_config_file generation_config.json \
--per_device_batch_size 8 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 2 \
--tag greedy \
--max_window_size 512 \
--max_new_tokens 256
```

# Stride = max_window_size

Tags: top-k-40, greedy, top-p-50


# Procedure

## Filtering
The dataset is filtered by the following criteria:
- The input is at least max_new_tokens + max_window_size long

## Window splitting
Given an input, it is first truncuated by max_new_tokens. The result is then split into window sizes of up to max_window_size. The minimum size of each window is l / math.ceil(l / max_size), where l is len(imput)-max_new_tokens. Each window is generated independently. The first window has index 0, the second has index 1, etc.

# AI Prompts

I want you to do automatic code generation. I will provide some code and you will attempt to autocomplete the code. You will respond with one line at a time.

