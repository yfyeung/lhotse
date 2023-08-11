import logging
import os
import json
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

LIBRILIGHT_FINETUNING = ("clean", "other")


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    file_name = str(audio_path).replace(".flac", "").replace(str(corpus_dir) + "/", "")
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    dir_path = os.path.dirname(audio_path)
    text_id = audio_path.stem
    dir_name = audio_path.stem[:-5]
    with open(dir_path + "/" + dir_name + ".trans.txt") as f:
        texts = f.readlines()
    for text in texts:
        if text_id in text:
            text = text[len(text_id) + 1:].strip()
            break

    recording = Recording.from_file(
        path=audio_path,
        recording_id=file_name,
    )
    segment = SupervisionSegment(
        id=file_name,
        recording_id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="English",
        text=text,
    )

    return recording, segment


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
    all_audio_paths = list(corpus_dir.rglob("*.flac"))
    audio_paths = [
        audio_path
        for audio_path in all_audio_paths if subset in str(audio_path)
    ]        

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
            futures.append(
                ex.submit(
                    _parse_utterance,
                    corpus_dir,
                    audio_path,
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

    return recording_set, supervision_set


def prepare_librilight_finetuning(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the LibriLight Finetuning dataset.
    :param books_dir: Path to the LibriLight Finetuning books.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir) if output_dir is not None else None

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing LibriLight Finetuning...")

    subsets = LIBRILIGHT_FINETUNING

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing LibriLight Finetuning subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="librilight_finetuning",
            suffix="jsonl.gz",
        ):
            logging.info(f"LibriLight Finetuning subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"librilight_finetuning_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"librilight_finetuning_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_librilight_finetuning("/star-xy/softwares/icefall_development/icefall_libri_light/egs/libri_heavy/ASR/download/librispeech_finetuning", ".", num_jobs=16)
