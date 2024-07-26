import os
import json
import datasets


_LANGUAGE_PAIRS = [("en", "de"), ("de", "en"), ("fr", "en"), ("en", "fr"), ("pt_br", "en"), ("en", "pt_br")]

_BASE_URL = os.getcwd()

_URL = {
    "train": {
        "en-de": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-de-en-train_sentence.json",
        "de-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-de-en-train_sentence.json",
        "fr-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-fr-en-train_sentence.json",
        "en-fr": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-fr-en-train_sentence.json",
        "pt_br-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-pt_br-en-train_sentence.json",
        "en-pt_br": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-pt_br-en-train_sentence.json",
    },
    "dev": {
        "en-de": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-de-en-dev_sentence.json",
        "de-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-de-en-dev_sentence.json",
        "fr-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-fr-en-dev_sentence.json",
        "en-fr": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-fr-en-dev_sentence.json",
        "pt_br-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-pt_br-en-dev_sentence.json",
        "en-pt_br": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-pt_br-en-dev_sentence.json",
    },
    "test": {
        "en-de": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-de-en-test_sentence.json",
        "de-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-de-en-test_sentence.json",
        "fr-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-fr-en-test_sentence.json",
        "en-fr": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-fr-en-test_sentence.json",
        "pt_br-en": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-pt_br-en-test_sentence.json",
        "en-pt_br": _BASE_URL + "/data//MAIA-sen-speaker/MAIA-pt_br-en-test_sentence.json",
    },
}

_DESCRIPTION = """"""
_HOMEPAGE = ""

_CITATION = """"""


class MultiAtisPlusPlus(datasets.GeneratorBasedBuilder):
    """MAIA Corpus"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"{l1}-{l2}", version=datasets.Version("1.0.0"), description=f"MAIA Corpus {l1}-{l2}"
        )
        for l1, l2 in _LANGUAGE_PAIRS
    ]

    def _info(self):
        src, tgt = self.config.name.split("-")
        features = datasets.Features(
            {
                "translation": {
                    "doc_id": datasets.Value("string"),
                    src: datasets.Value("string"),
                    tgt: datasets.Value("string"),
                }
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang_pair = tuple(self.config.name.split("-"))
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "lang_pair": lang_pair,
                    "filepath": data_dir["train"][self.config.name],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "lang_pair": lang_pair,
                    "filepath": data_dir["test"][self.config.name],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "lang_pair": lang_pair,
                    "filepath": data_dir["dev"][self.config.name],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, lang_pair, filepath, split):
        """Yields examples."""
        src, target = lang_pair
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, sentence in enumerate(data):
                yield idx, {
                    "translation": {
                        "doc_id": sentence["doc_id"],
                        src: sentence[src],
                        target: sentence[target],
                    }
                }
