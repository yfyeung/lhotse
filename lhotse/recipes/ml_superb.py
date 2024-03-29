"""
About the ml_superb corpus

Libri-light is a benchmark for the training of automatic speech recognition (ASR)
systems with limited or no supervision.

It contains a large dataset of 60K hours of unlabelled speech from audiobooks in
English and a small labelled dataset (10h, 1h, and 10 min) plus metrics,
trainable baseline models, and pretrained models that use these datasets.

It is covered in more detail at https://arxiv.org/abs/1912.07875

This data is very huge - please download manually at ML_SUPERB_URL.
"""

import json
import logging
import os
import re
import string
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

ML_SUPERB = (
    "1h-train",
    # "10min-train",
    "dev",
    "test",
)

ML_SUPERB_MAP = {
    "ALFFA": "ALFFA",
    "cv": "commonvoice",
    "fleurs": "fleurs",
    "googlei18n-asr": "googlei18n_asr",
    "googlei18n-tts": "googlei18n_tts",
    "LAD": "LAD",
    "M-AILABS": "M-AILABS",
    "mexico-el": "mexico-el",
    "mls": "mls",
    "nchlt": "nchlt",
    "NST": "NST",
    "swc": "swc",
    "voxforge": "voxforge",
    "voxpopuli": "voxpopuli",
}


def _normalize(
    old: str,
) -> str:
    punctuation = string.punctuation
    additional_chars = "≠×£¥€،؛؟ـ«»¡¿·‘’“”§٪…°⅛½¾─—" + " \n\r\t"
    all_chars_to_remove = punctuation + additional_chars
    regex_pattern = f"[{re.escape(all_chars_to_remove)}]"
    return re.sub(regex_pattern, "", old)


def _parse_utterance(
    corpus_dir: Pathlike,
    file_name: str,
    audio_path: Pathlike,
    text: str,
    discrete_tokens: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=file_name,
    )

    segment = SupervisionSegment(
        id=file_name + "_sp1",
        recording_id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=file_name.split("_", 2)[1],
        text=_normalize(text),
    )
    segment.discrete_tokens = discrete_tokens

    return recording, segment


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    discrete_tokens_info: dict,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    transcript_paths = [
        path
        for path in list(corpus_dir.rglob("*.txt"))
        if subset.replace("-", "_") in str(path)
    ]
    audio_infos = []
    for transcript_path in transcript_paths:
        with open(transcript_path) as f:
            audio_infos.extend(f.read().splitlines())
    audio_infos = set(audio_infos)

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_info in tqdm(audio_infos, desc="Distributing tasks"):
            if len(audio_info) == 0:
                continue
            file_name, _, text = audio_info.replace("\t", " ").split(" ", 2)
            if file_name not in discrete_tokens_info.keys():
                logging.info(f"Skip {file_name}")
                continue
            discrete_tokens = discrete_tokens_info[file_name]
            audio_path = file_name.split("_", 2)
            audio_path = (
                corpus_dir
                / ML_SUPERB_MAP[audio_path[0]]
                / audio_path[1]
                / "wav"
                / (file_name + ".wav")
            )
            audio_path = audio_path.resolve()
            futures.append(
                ex.submit(
                    _parse_utterance,
                    corpus_dir,
                    file_name,
                    audio_path,
                    text,
                    discrete_tokens,
                )
            )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_ml_superb(
    corpus_dir: Pathlike,
    discrete_token_path: Pathlike,
    output_dir: Optional[Pathlike] = None,
    duration: str = "1h",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the ML-SUPERB dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    discrete_token_path = Path(discrete_token_path)
    assert discrete_token_path.is_file(), f"No such file: {discrete_token_path}"

    logging.info("Preparing ML-SUPERB...")

    subsets = ML_SUPERB

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    discrete_tokens_info = {}
    with open(discrete_token_path) as f:
        discrete_tokens = f.read().splitlines()
        for discrete_token in discrete_tokens:
            discrete_token = discrete_token.split(" ", 1)
            discrete_tokens_info[discrete_token[0]] = discrete_token[1]

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing ML-SUPERB subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="ml_superb",
            suffix="jsonl.gz",
        ):
            logging.info(f"ML-SUPERB subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(
            part, corpus_dir, discrete_tokens_info, num_jobs
        )

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"ml_superb_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"ml_superb_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    prepare_ml_superb(
        "/mnt/cloudstorfs/sjtu_home/yifan.yang02/raw_data/ml_superb",
        "/mnt/cloudstorfs/sjtu_home/yifan.yang02/audio-discretizer/data/ml_superb_whisper_large_v3_kms2000/out_quantized",
        ".",
        num_jobs=16,
    )
