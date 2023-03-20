import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, GenerationConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import os
import urllib.parse
import numpy as np
import logging
import random
from accelerate.logging import get_logger
get_logger("transformers").setLevel(logging.ERROR)
logger = get_logger(__name__)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Do inference with a transformer model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="The name of the split to use.",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of the dataset to use.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--generation_config_file",
        type=str,
        default=None,
        help="Generation config path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_batch_size",  # "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached dataset"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="The tag to use for this generation run."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="The maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--max_window_size",
        type=int,
        default=None,
        help="The maximum number of tokens in the input."
    )
    parser.add_argument(
        "--subsamples",
        type=int,
        default=None,
        help="The number of subsamples to use from each data example. None means use all."
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need a dataset name.")
    
    return args


def main():
    """
    Generate new data by sampling from the original data.
    """

    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    if args.seed is not None:
        set_seed(args.seed)


    save_dir = Path(args.output_dir, args.dataset_name, args.model_name_or_path, args.dataset_config_name, args.tag)
    # Write the generation config to disk
    if accelerator.is_main_process:
        if args.output_dir is not None:
            
            #safe_dataset_name = urllib.parse.quote(args.dataset_name, safe='')
            #urlencode args.dataset_name
            #safe_model_name = urllib.parse.quote(args.model_name_or_path, safe='')
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Need a output directory.")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # write args to disk
        with open( save_dir / "args.json", "w") as f:
            json.dump(args.__dict__, f, indent=4)
    

        # 
    # Load the dataset
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # if args.dataset_name is not None:
    #    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    #    dataset = raw_datasets.with_format("torch", columns=[text_column])

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split)

        indices = list(range(len(raw_dataset)))
        raw_dataset = raw_dataset.add_column("id", indices)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    # model = model.to(accelerator.device) # TODO: check if this is needed

    generation_config = {}
    if args.generation_config_file is not None:
        # read from file
        with open(args.generation_config_file, "r") as f:
            generation_config = json.load(f)
            generation_config = GenerationConfig.from_dict(generation_config)


    elif args.model_name_or_path:
        generation_config = model.generation_config
        logger.warning(f"Using default generation config from model: {generation_config}")
    
    if args.max_new_tokens is not None:
        generation_config.max_new_tokens = args.max_new_tokens

    max_new_tokens = generation_config.max_new_tokens
    if max_new_tokens == None:
        raise ValueError("max_new_tokens is not set in the generation config.")
    else:
        logger.info(f"max_new_tokens are set to {max_new_tokens}")


    # Write the model config and generation config to disk
    if accelerator.is_main_process:
        print(generation_config)

        # Dump the model config without defaults to disk
        with open( save_dir.parent.parent / "model_config_diff.json", "w") as f:
            json.dump(config.to_diff_dict(), f, indent=4)

        # Dump the model config with defaults to disk
        with open(save_dir.parent.parent / "model_config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=4)

        # Dump the generation config without defaults to disk
        with open(save_dir / "generation_config_diff.json", "w") as f:
            json.dump(generation_config.to_diff_dict(), f, indent=4)

        # Dump the generation config with defaults to disk
        with open(save_dir / "generation_config.json", "w") as f:
            json.dump(generation_config.to_dict(), f, indent=4)


    # Preprocessing the datasets.
    column_names = raw_dataset.column_names

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]
        logger.warning(f"Using column {text_column_name} as text column.")


    max_input_length = args.max_window_size + max_new_tokens
    if max_input_length > model.config.max_position_embeddings:
        raise ValueError(
            f"max_window_size ({args.max_window_size}) + max_new_tokens ({max_new_tokens}) is larger than the maximum position embedding size "
            f"({model.config.max_position_embeddings})."
        )
    
    min_input_length = args.max_window_size + max_new_tokens # TODO: check if this is a good value
    
    # Tokenize the data
    def tokenize_function(examples):        
        return tokenizer(examples[text_column_name])
    
    def filter_function(examples):
        res = []
        for example in examples["input_ids"]:
            if len(example) < min_input_length:
                res.append(False)
            else:
                res.append(True)
        return res
    
    def array_chunk_min_max(array, min_size, max_size):
        l = len(array)
        chunks, remainder = divmod(l, max_size)
        if remainder > 0:
            chunks += 1
            # Increased chunks to "chunks"
            if (l//chunks) < min_size:
                # Reduced chunks to "chunks" because "l//chunks" < min_size
                chunks -= 1
                # Truncating remainder
                array = array[:chunks*max_size]
                
            return np.array_split(array, chunks)
        else:
            return np.array_split(array, chunks)
    

    def array_chunk_max(array, max_size):
        """
        Splits an array into chunks of maximum size max_size.
        The minimum size of the resulting chunks is l / math.ceil(l / max_size).
        """
        chunks, remainder = divmod(len(array), max_size)
        if remainder > 0:
            chunks += 1
        return np.array_split(array, chunks)



    def minibatch_function(examples, indices):
        new_examples = {
            "id": [],
            "part": [],
            "input_ids": [],
            "attention_mask": [],
            "reference_input_ids": [],
        }

        for i, id in enumerate(indices):
            input_ids = examples["input_ids"][i][:-max_new_tokens]
            mask = examples["attention_mask"][i][:-max_new_tokens]
            
            minibatch_ids = array_chunk_max(input_ids, args.max_window_size)
            minibatch_mask = array_chunk_max(mask, args.max_window_size)
            
            reference_input_ids = minibatch_ids[1:]
            end_ids = examples["input_ids"][i][-max_new_tokens:]
            reference_input_ids.append(end_ids)

            minibatch_size = len(minibatch_ids)
            sample_size = minibatch_size
            if args.subsamples is not None:
                sample_size = min(args.subsamples, minibatch_size)

            sample_indices = sorted(random.sample(range(minibatch_size), sample_size))
            minibatch_ids = [minibatch_ids[i] for i in sample_indices]
            minibatch_mask = [minibatch_mask[i] for i in sample_indices]
            reference_input_ids = [reference_input_ids[i] for i in sample_indices]
            

            new_examples["id"].extend([id]*sample_size)
            new_examples["part"].extend(list(zip(sample_indices, [minibatch_size]*sample_size)))
            new_examples["input_ids"].extend(minibatch_ids)
            new_examples["attention_mask"].extend(minibatch_mask)
            new_examples["reference_input_ids"].extend(reference_input_ids)
        return new_examples


    with accelerator.main_process_first():
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        filtered_dataset = tokenized_dataset.filter(
            filter_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Filtering min length",
        )
        minibatch_dataset = filtered_dataset.map(
            minibatch_function,
            with_indices=True,
            batched=True,
            remove_columns=tokenized_dataset.column_names,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Splitting records into minibatches",
        )
        #dataset = tokenized_datasets.with_format("torch", columns=[text_column], output_all_columns=True)

    dataset = minibatch_dataset
    
    def data_collator(examples):
        batch = tokenizer.pad(examples)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])

        return batch

    # Create the DataLoader
    data_loader = DataLoader(dataset, shuffle=False,
                             collate_fn=data_collator, batch_size=args.per_device_batch_size)

    # Prepare everything with `accelerator`.
    model, data_loader = accelerator.prepare(model, data_loader)
    model = accelerator.unwrap_model(model)
    #model, data_loader = accelerator.prepare(
    #    model, data_loader, device_placement=[True, False])

    # save the data
    i = "{:05n}".format(accelerator.process_index + 1)
    n = "{:05n}".format(accelerator.num_processes)

    path = save_dir / (f"{i}-of-{n}" + f".{args.dataset_split}.jsonl")
    fp = open(path, 'w')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(data_loader) * args.per_device_batch_size), position=accelerator.process_index) #,disable=not accelerator.is_local_main_process)
    i = 0
    for batch in data_loader:
        #if i < 28:
        #   i += 1
        #   continue

        prompt_ids = batch["input_ids"]
        prompt_ids.to(accelerator.device)
        attention_mask = batch["attention_mask"]
        attention_mask.to(accelerator.device)

        # accelerator.print("Generating...")
        with torch.no_grad():
            # generate the data

            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                #max_new_tokens=args.max_new_tokens,
            )

        # decode the data
        decoded_prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        predicted_ids = generated[:, -max_new_tokens:]
        decoded_predictions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        reference_ids = batch["reference_input_ids"]
        decoded_reference = tokenizer.batch_decode(reference_ids, skip_special_tokens=True)

        progress_bar.update(args.per_device_batch_size)

        # save the data to disk
        for index in range(generated.shape[0]):
            # print("saving..."):
            #colnames = batch.keys()
            #entry = {colname: batch[colname][index] for colname in colnames}
            entry = {"id": batch["id"][index], "part": batch["part"][index]}
            #entry = {"text": batch[text_column_name][index]}
            #entry.pop('input_ids', None)
            #entry.pop('attention_mask', None)
            entry["prompt"] = decoded_prompts[index]
            entry["reference"] = decoded_reference[index]
            entry["prediction"] = decoded_predictions[index]
            entry["ended"] = predicted_ids[index][-1].item() == tokenizer.eos_token_id
            fp.write(json.dumps(entry) + "\n")
            fp.flush()

    fp.close()

    accelerator.wait_for_everyone()
    
    progress_bar.close()


if __name__ == "__main__":
    main()
