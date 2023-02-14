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
--dataset_name code_search_net \
--dataset_config_name python \
--dataset_split test \
--text_column_name func_documentation_string \
--model_name_or_path gpt2 \
--per_device_batch_size 8 \
--output_dir output \
--seed 42 \
--preprocessing_num_workers 10 \ 
--tag K40 \
--max_new_tokens 100
```