import json
import logging
import os
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations

VI2000 = ("test", "dev", "train")


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_info: dict,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    audio_id = audio_info["key"]
    audio_path = corpus_dir / audio_info["wav"]
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=audio_id,
    )

    supervision = SupervisionSegment(
        id=audio_id,
        recording_id=audio_id,
        start=0,
        duration=recording.duration,
        channel=0,
        text=audio_info["txt"].strip(),
        language="Vietnamese",
    )

    return recording, supervision


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    metadata_path = corpus_dir / f"{subset}.list"
    with open(metadata_path) as f, ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for line in tqdm(f, desc="Distributing tasks"):
            audio_info = json.loads(line)
            futures.append(ex.submit(_parse_utterance, corpus_dir, audio_info))

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recording, supervision = result
            recordings.append(recording)
            supervisions.append(supervision)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_vi2000(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the vi2000 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing vi2000...")

    subsets = VI2000

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing vi2000 subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="vi2000",
            suffix="jsonl.gz",
        ):
            logging.info(f"vi2000 subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(output_dir / f"vi2000_supervisions_{part}.jsonl.gz")
            recording_set.to_file(output_dir / f"vi2000_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
