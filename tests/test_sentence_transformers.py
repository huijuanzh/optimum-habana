import csv
import gzip
import os

import pytest
from sentence_transformers import SentenceTransformer, util

from optimum.habana.utils import HabanaGenerationTime

from .test_examples import TIME_PERF_FACTOR
from .utils import OH_DEVICE_CONTEXT


MODELS_TO_TEST = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/multi-qa-distilbert-cos-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-albert-small-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
]


def _test_sentence_transformers(
    model_name: str,
    baseline,
):
    model = SentenceTransformer(model_name)

    nli_dataset_path = "/tmp/datasets/AllNLI.tsv.gz"
    sentences = set()
    max_sentences = 10000

    # Download datasets if needed
    if not os.path.exists(nli_dataset_path):
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            sentences.add(row["sentence1"])
            if len(sentences) >= max_sentences:
                break

    sentences = list(sentences)

    measured_throughput0 = []
    for j in range(10):
        for i in range(2):
            with HabanaGenerationTime() as timer:
                _ = model.encode(sentences, batch_size=32)
            diff_time = timer.last_duration
        measured_throughput0.append(len(sentences) / diff_time)
    measured_throughput0.sort()
    measured_throughput = sum(measured_throughput0[2:8]) / 6

    # Only assert the last measured throughtput as the first iteration is used as a warmup
    baseline.assertRef(
        compare=lambda actual, ref: actual >= (2 - TIME_PERF_FACTOR) * ref,
        context=[OH_DEVICE_CONTEXT],
        measured_throughput=measured_throughput,
    )


@pytest.mark.parametrize("model_name", MODELS_TO_TEST)
def test_compute_embeddings_throughput(model_name: str, baseline):
    _test_sentence_transformers(model_name, baseline)
