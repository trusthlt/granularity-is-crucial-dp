from datasets import load_dataset
from typing import Any
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"


class Preprocessor:
    """
    Description
    -----------
    Abstract class based on which built-in and custom preprocessors are
    prepared.
    Attributes
    ----------
    input_max_seq_len : int
    Methods
    -------
    process_data():
        Given a loaded dataset from HF or local path, output the preprocessed
        dataset.
    """

    def __init__(self, model, tokenizer, lang_pair, input_max_seq_len):
        self.model = model
        self.tokenizer = tokenizer
        self.lang_pair = lang_pair
        self.source, self.target = lang_pair.split('-')
        self.input_max_seq_len = input_max_seq_len
        self.train_size = 0

    def shift_tokens_right(self, input_ids, pad_token_id: int, decoder_start_token_id=None) -> Any:
        pass

    def preprocess_function(self, examples):
        pass

    def process_data(
            self,
            data_name: str,
            num_train_examples=None,
            num_eval_examples=None,
            eval_only: bool = False,
            test_only: bool = False
    ):
        """
        Description
        -----------
        Given a loaded dataset from HF or local path, output the preprocessed
        dataset.
        Parameters
        ----------
        :param data_name:
        :param num_train_examples:
        :param num_eval_examples:
        :param eval_only:
        :param test_only:
        """
        if test_only:
            test_dataset: Any = load_dataset(data_name, self.lang_pair, split='test')
            test_dataset = test_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=test_dataset.column_names
            )
            return test_dataset
        else:
            if num_eval_examples is None:
                eval_dataset: Any = load_dataset(data_name, self.lang_pair, split='validation')
            else:
                eval_dataset: Any = load_dataset(
                    data_name,
                    self.lang_pair,
                    split='validation'
                ).select(range(num_eval_examples))
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
            if not eval_only:
                if num_train_examples is None:
                    train_dataset: Any = load_dataset(data_name, self.lang_pair, split='train')
                else:
                    train_dataset: Any = load_dataset(
                        data_name,
                        self.lang_pair,
                        split='train'
                    ).select(range(num_train_examples))

                self.train_size = train_dataset.num_rows

                train_dataset = train_dataset.map(
                    self.preprocess_function,
                    batched=True,
                    remove_columns=train_dataset.column_names
                )

                return train_dataset, eval_dataset
            else:
                return eval_dataset


class T5Preprocessor(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_prefix = T5Preprocessor.prepare_t5_prefix(self.lang_pair)

    def shift_tokens_right(self, input_ids, pad_token_id: int, decoder_start_token_id=None) -> np.ndarray:
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = np.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id

        shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
        return shifted_input_ids

    def preprocess_function(self, examples):
        # We only add it to the input data according to 
        # https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py
        inputs = [self.task_prefix + example[self.source] for example in examples["translation"]]
        targets = [example[self.target] for example in examples["translation"]]
        model_inputs = self.tokenizer(
            inputs, max_length=self.input_max_seq_len, padding="max_length", truncation=True, return_tensors="np"
        )
        # Set up the tokenizer for targets
        labels = self.tokenizer(
            targets,
            max_length=self.input_max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        model_inputs["labels"] = labels["input_ids"]
        decoder_input_ids = self.shift_tokens_right(
            labels["input_ids"],
            self.model.config.pad_token_id,
            self.model.config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

        # We need decoder_attention_mask, so we can ignore pad tokens from loss
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs

    @staticmethod
    def prepare_t5_prefix(lang):
        src, tgt = lang.split('-')
        special_lang = {
            "de": "German",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "it": "Italian",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ar": "Arabic",
            "cs": "Czech",
            "el": "Greek",
            "hi": "Hindi",
            "ro": "Romanian",
            "ja": "Japanese",
        }
        try:
            prefix = "translate {} to {}: ".format(special_lang[src], special_lang[tgt])
        except KeyError:
            prefix = ""
        return prefix


class MT5Preprocessor(T5Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_prefix = ""


class MLongT5Preprocessor(MT5Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
