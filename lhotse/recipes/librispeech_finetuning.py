import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

LIBRISPEECH_FINETUNING = ("1h", "9h")

LIBRISPEECH_FINETUNING_URL = (
    "https://dl.fbaipublicfiles.com/librispeech_finetuning/data/librispeech_finetuning.tgz",
)


def parse_utterance(
    dataset_split_path: Path,
    line: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # Create the Recording first
    audio_path = (
        dataset_split_path
        / Path(recording_id.replace("-", "/")).parent
        / f"{recording_id}.flac"
    )
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="English",
        speaker=re.sub(r"-.*", r"", recording.id),
        text=text.strip(),
    )
    return recording, segment


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    normalize_text: str = "none",
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

    with ProcessPoolExecutor(num_jobs) as ex:
        recordings = []
        supervisions = []
        futures = []
        for trans_path in tqdm(
            part_path.rglob("*.trans.txt"), desc="Distributing tasks", leave=False
        ):
            # "trans_path" file contains lines like:
            #
            #   121-121726-0000 ALSO A POPULAR CONTRIVANCE
            #   121-121726-0001 HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE
            #   121-121726-0002 ANGOR PAIN PAINFUL TO HEAR
            #
            # We will create a separate Recording and SupervisionSegment for those.
            with open(trans_path) as f:
                for line in f:
                    futures.append(ex.submit(parse_utterance, part_path, line))

        for future in tqdm(futures, desc="Processing", leave=False):
            result = future.result()
            if result is None:
                continue
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Normalize text to lowercase
        if normalize_text == "lower":
            to_lower = lambda text: text.lower()
            supervision_set = SupervisionSet.from_segments(
                [s.transform_text(to_lower) for s in supervision_set]
            )

        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_librispeech_finetuning(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the LibriSpeech Finetuning dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing LibriSpeech Finetuning...")

    subsets = LIBRISPEECH_FINETUNING

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing LibriSpeech Finetuning subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="librispeech_finetuning",
            suffix="jsonl.gz",
        ):
            logging.info(
                f"LibriSpeech Finetuning subset: {part} already prepared - skipping."
            )
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"librispeech_finetuning_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"librispeech_finetuning_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_librispeech_finetuning(
        "/private_data2/librispeech_finetuning", ".", num_jobs=16
    )
