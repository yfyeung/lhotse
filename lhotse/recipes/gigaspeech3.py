import logging
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import load_manifest
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

GIGASPEECH3_URL = "https://huggingface.co/datasets/speechcolab/gigaspeech3"

GIGASPEECH3 = ("test", "ytnnews24-0", "ytnnews24-1")


def _parse_utterance(
    part_dir: Pathlike,
    line: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    segment_id, text = line.split(" ", 1)
    audio_path = part_dir / segment_id.rsplit("-", 1)[0] / f"{segment_id}.wav"
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=segment_id,
    )

    segment = SupervisionSegment(
        id=segment_id,
        recording_id=segment_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="Korean",
        text=text.strip(),
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
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / subset
    trans_paths = list(part_path.rglob("*.trans.txt"))

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for trans_path in tqdm(trans_paths, desc="Distributing tasks"):
            with open(trans_path) as f:
                for line in f:
                    futures.append(ex.submit(_parse_utterance, part_path, line))

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


def prepare_gigaspeech3(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the GigaSpeech 3 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing GigaSpeech 3...")

    subsets = GIGASPEECH3

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing GigaSpeech 3 subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="gigaspeech3",
            suffix="jsonl.gz",
        ):
            logging.info(f"GigaSpeech 3 subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"gigaspeech3_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"gigaspeech3_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_gigaspeech3("/scratch/GigaSpeech3", ".", num_jobs=1)
