"""TODO(IndicSentiment): Add a description here."""


import json

import datasets


_HOMEPAGE = ""

_CITATION = """\

"""

_DESCRIPTION = """\

"""

_LANG = ["as", "bn", "bd", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
_URL = "https://huggingface.co/datasets/ai4bharat/IndicSentiment/resolve/main/data/{split}/{language}.json"
_VERSION = datasets.Version("1.0.0", "First version of IndicSentiment")


class HAIndicSentiment(datasets.GeneratorBasedBuilder):
    """TODO(IndicSentiment): Short description of my dataset."""
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"translation-{lang}",
            description=f"translated sentiment data for {lang}",
            version=_VERSION,
        )
        for lang in _LANG
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION + self.config.description,
            features=datasets.Features(
                {
                    "GENERIC CATEGORIES": datasets.Value("string"),
                    "CATEGORY": datasets.Value("string"),
                    "SUB-CATEGORY": datasets.Value("string"),
                    "PRODUCT": datasets.Value("string"),
                    "BRAND": datasets.Value("string"),
                    "ASPECTS": datasets.Value("string"),
                    "ASPECT COMBO": datasets.Value("string"),
                    "ENGLISH REVIEW": datasets.Value("string"),
                    "LABEL": datasets.Value("string"),
                    "INDIC REVIEW": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        *translation_prefix, language = self.config.name.split("-")
        splits = {datasets.Split.VALIDATION: "validation", datasets.Split.TEST: "test"}

        data_urls = {
            split: _URL.format(language=language, split=splits[split]) for split in splits
        }
        dl_paths = dl_manager.download(data_urls)
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": dl_paths[split]},
            )
            for split in splits
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""

        with open(filepath, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                data = json.loads(row)
                yield idx, data
