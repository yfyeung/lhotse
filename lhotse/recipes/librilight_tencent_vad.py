"""
About the librilight corpus

Libri-light is a benchmark for the training of automatic speech recognition (ASR)
systems with limited or no supervision.

It contains a large dataset of 60K hours of unlabelled speech from audiobooks in
English and a small labelled dataset (10h, 1h, and 10 min) plus metrics,
trainable baseline models, and pretrained models that use these datasets.

It is covered in more detail at https://arxiv.org/abs/1912.07875

This data is very huge - please download manually at LIBRILIGHT_URL.
"""

import logging
import os
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations

LIBRILIGHT = ("small", "medium", "large")

LIBRILIGHT_URL = (
    "https://dl.fbaipublicfiles.com/librilight/data/small.tar",
    "https://dl.fbaipublicfiles.com/librilight/data/medium.tar",
    "https://dl.fbaipublicfiles.com/librilight/data/large.tar",
)


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
    segment_infos: Optional[list] = None,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    file_name = str(audio_path).replace(".flac", "").replace(str(corpus_dir) + "/", "")
    speaker = str(audio_path).split("/")[-3]
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=file_name,
    )

    segments = []
    if segment_infos is not None:
        for segment_info in segment_infos:
            segments.append(
                SupervisionSegment(
                    id=segment_info[0],
                    recording_id=file_name,
                    start=segment_info[1],
                    duration=add_durations(
                        segment_info[2], -segment_info[1], sampling_rate=16000,
                    ),
                    channel=0,
                    language="English",
                    speaker=speaker,
                )
            )
    else:
        segments.append(
            SupervisionSegment(
                id=file_name,
                recording_id=file_name,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="English",
                speaker=speaker,
            )
        )

    return recording, segments


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    vad_info_dict: Optional[defaultdict] = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / subset
    audio_paths = list(part_path.rglob("*.flac"))

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
            if vad_info_dict is not None:
                segment_infos = vad_info_dict[
                    str(audio_path).split("/", -1)[-1].replace(".flac", "")
                ]
            futures.append(
                ex.submit(_parse_utterance, corpus_dir, audio_path, segment_infos)
            )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recording, segments = result
            recordings.append(recording)
            supervisions.extend(segments)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_librilight(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    vad_path: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the LibriLight dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing LibriLight...")

    subsets = LIBRILIGHT

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if vad_path is not None:
        logging.info(f"Using Vad from {vad_path}")
        vad_path = Path(vad_path)
        with open(vad_path) as f:
            vad_infos = f.read().splitlines()
        vad_info_dict = defaultdict(list)
        for vad_info in vad_infos:
            vad_info = vad_info.split(" ", -1)
            vad_info_dict[vad_info[1]].append(
                (vad_info[0], float(vad_info[2]), float(vad_info[3]))
            )

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing LibriLight subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="librilight",
            suffix="jsonl.gz",
        ):
            logging.info(f"LibriLight subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(
            part, corpus_dir, vad_info_dict, num_jobs
        )

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"librilight_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"librilight_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_librilight(
        "/data/LibriLight",
        ".",
        "/data/yfy62/librilight_demo/download/segments",
        num_jobs=16,
    )
