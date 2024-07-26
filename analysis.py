import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "agemagician/mlong-t5-tglobal-base",
    use_fast=False,
    legacy=False
)


def get_corpus_stats(dataset, source, target):
    print(dataset)
    for part in ["train", "validation", "test"]:
        len_all_source = []
        len_all_target = []
        max_length_source = 0
        max_length_target = 0
        for file in tqdm(dataset[part]["translation"]):
            source_doc = file[source]
            target_doc = file[target]
            len_source = len(tokenizer.tokenize(source_doc))
            len_target = len(tokenizer.tokenize(target_doc))
            len_all_source.append(len_source)
            len_all_target.append(len_target)
            if len_source > max_length_source:
                max_length_source = len_source
            if len_target > max_length_target:
                max_length_target = len_target
        print(part)
        print(f"Average length {source}: ", sum(len_all_source)/len(len_all_source))
        print(f"Average length {target}: ", sum(len_all_target)/len(len_all_target))
        print(f"Standard deviation {source}: ", np.std(len_all_source))
        print(f"Standard deviation {target}: ", np.std(len_all_target))
        print(f"Max length {source}", max_length_source)
        print(f"Max length {target}", max_length_target)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data', type=str, required=True)
    arg_parser.add_argument('--input_lang', type=str, required=True)
    arg_parser.add_argument('--output_lang', type=str, default='en')
    args = arg_parser.parse_args()
    # Load data
    data = load_dataset(args.data, f"{args.input_lang}-{args.output_lang}")
    # Get corpus statistics
    get_corpus_stats(data, args.input_lang, args.output_lang)