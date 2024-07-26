from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM
from preprocessing import MLongT5Preprocessor
from torch.utils.data import DataLoader
from flax.training import train_state
from datetime import datetime
import jax.numpy as jnp
from typing import Any
from tqdm import tqdm
import evaluate
import logging
import optax
import json
import glob
import jax
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AttackExperiment:
    def __init__(self, settings) -> None:
        self.seed = settings.seed
        self.dataset_infer = settings.dataset_infer
        self.dataset_loss = settings.dataset_loss
        self.base_path: str = "/".join(settings.base_model.split("/")[:-1])
        self.target_path = "/".join(settings.target_model.split("/")[:-1])
        self.base_model = FlaxAutoModelForSeq2SeqLM.from_pretrained(settings.base_model, seed=self.seed)
        self.target_model = FlaxAutoModelForSeq2SeqLM.from_pretrained(settings.target_model, seed=self.seed)
        self.batch_size = settings.batch_size
        self.dataset_infer_name = self.dataset_infer.split("/")[-1].split(".")[0]
        self.dataset_loss_name = self.dataset_loss.split("/")[-1].split(".")[0]
        # noinspection PyTypeChecker
        self.metrics = evaluate.combine([
            "accuracy", "f1", "precision","recall", "ealvaradob/false_positive_rate",
        ])

        self.threshold = self._base_threshold()
        self.attack_writer = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            'agemagician/mlong-t5-tglobal-base',
            legacy=False,
            use_fast=False
        )
        self.labels = []

        self.preprocessor = MLongT5Preprocessor(
            model=self.base_model,
            tokenizer=self.tokenizer,
            lang_pair=settings.lang_pair,
            input_max_seq_len=settings.input_max_seq_len,
        )
        self._setup_dataloader(settings)

    def _setup_dataloader(self, settings) -> None:
        train_dataset_loss, _ = self.preprocessor.process_data(
            data_name=settings.dataset_loss,
        )

        self.train_data_loader_loss = DataLoader(
            train_dataset_loss,
            batch_size=self.batch_size,
            collate_fn=lambda batch: {key: jnp.array([sample[key] for sample in batch]) for key in batch[0].keys()},
        )

        train_dataset, eval_dataset = self.preprocessor.process_data(
            data_name=settings.dataset_infer,
        )

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: {key: jnp.array([sample[key] for sample in batch]) for key in batch[0].keys()},
        )

        self.eval_data_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: {key: jnp.array([sample[key] for sample in batch]) for key in batch[0].keys()},
        )

        test_dataset =  self.preprocessor.process_data(
            data_name=settings.dataset_infer,
            test_only=True,
        )

        self.test_data_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda batch: {key: jnp.array([sample[key] for sample in batch]) for key in batch[0].keys()},
        )

    def _base_threshold(self):
        threshold = sorted(glob.glob(f'{self.base_path}/threshold_{self.dataset_loss_name}*'))
        if len(threshold) == 0:
            return 0
        else:
            with open(threshold[-1], "r") as f:
                return float(f.read())

    def eval_step(self, state, batch):
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=state.params, train=False).logits
        loss = self.loss_function(logits, labels)
        return loss

    def loss_function(self, logits, labels):
        cross_entropy = optax.softmax_cross_entropy(
            logits,
            jax.nn.one_hot(labels, num_classes=self.base_model.config.vocab_size))
        return jnp.mean(cross_entropy, axis=1)

    def run_experiment(self) -> None:
        eval_step = jax.jit(self.eval_step)
        if self.threshold == 0:
            base_state: train_state.TrainState = train_state.TrainState.create(
                apply_fn=self.base_model.__call__,
                params=self.base_model.params,
                tx=optax.adam(0.0001),
            )
            base_train_loss = []
            for _, batch in tqdm(enumerate(self.train_data_loader_loss), desc="Loss of training data on base model ...."):
                eval_metrics = eval_step(base_state, batch)
                base_train_loss.extend(eval_metrics)
            self.threshold = jnp.mean(jnp.array(base_train_loss))
            with open(f'{self.base_path}/threshold_{self.dataset_loss_name}.txt', "w") as f:
                f.write(str(self.threshold))

        target_state: train_state.TrainState = train_state.TrainState.create(
            apply_fn=self.target_model.__call__,
            params=self.target_model.params,
            tx=optax.adam(0.0001),
        )
        target_train_loss = []
        for _, batch in tqdm(enumerate(self.train_data_loader), desc="Loss of training data on target model ...."):
            self.labels.extend(self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
            eval_metrics = eval_step(target_state, batch)
            target_train_loss.extend(eval_metrics)

        target_dev_loss = []
        for _, batch in tqdm(enumerate(self.eval_data_loader), desc="Loss of dev data on target model ...."):
            eval_metrics = eval_step(target_state, batch)
            target_dev_loss.extend(eval_metrics)

        target_test_loss = []
        for _, batch in tqdm(enumerate(self.test_data_loader), desc="Loss of test data on target model ...."):
            eval_metrics = eval_step(target_state, batch)
            target_test_loss.extend(eval_metrics)

        print("threshold", self.threshold)
        train_label: list[int] = [1] * len(target_train_loss)
        dev_label: list[int] = [0] * len(target_dev_loss)
        test_label: list[int] = [0] * len(target_test_loss)
        prediction_train: list[int] = [1 if loss < self.threshold else 0 for loss in target_train_loss]
        prediction_dev: list[int] = [1 if loss < self.threshold else 0 for loss in target_dev_loss]
        prediction_test: list[int] = [1 if loss < self.threshold else 0 for loss in target_test_loss]
        label = train_label + dev_label + test_label
        prediction = prediction_train + prediction_dev + prediction_test
        for ref, pred in zip(label, prediction):
            self.metrics.add(references=ref, predictions=pred)
        results = self.metrics.compute()
        print("seed", self.seed)
        print("results", results)
        leakage = results["recall"] - results["false_positive_rate"]
        print("leakage", leakage)

        self.attack_writer["dataset_loss"] = self.dataset_loss_name
        self.attack_writer["dataset_infer"] = self.dataset_infer_name
        self.attack_writer["threshold"] = float(self.threshold)
        self.attack_writer["results"] = results
        self.attack_writer["leakage"] = leakage
        self.attack_writer["leaked"] = []
        for pred, label in zip(prediction, self.labels):
            if pred == 1:
                self.attack_writer["leaked"].append(label)

        with open(f'{self.target_path}/attack_{self.dataset_infer_name}_result.json', "w") as f:
            json.dump(self.attack_writer, f, indent=4, separators=(',', ': '))