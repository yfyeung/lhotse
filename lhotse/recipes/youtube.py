import logging
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


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
    language: str = None,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    file_name = audio_path.stem
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=file_name,
    )

    supervision = SupervisionSegment(
        id=file_name,
        recording_id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
    )

    return recording, supervision


def _prepare_subset(
    subset: str,
    corpus_dir: Pathlike,
    language: str = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param subset: str, the name of the subset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param language: str of the language of corpus.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    corpus_dir = Path(corpus_dir)
    part_path = corpus_dir / subset
    audio_paths = list(part_path.rglob("*.wav"))

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
            futures.append(
                ex.submit(_parse_utterance, corpus_dir, audio_path, language)
            )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recordings.append(result[0])
            supervisions.append(result[1])

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_youtube(
    corpus_name: str,
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: Optional[str] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the youtube dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param language: str of the language of corpus.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info(f"Preparing {corpus_name}...")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(corpus_dir.iterdir(), desc="Dataset parts"):
        part = part.stem
        logging.info(f"Processing {corpus_name} subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix=corpus_name,
            suffix="jsonl.gz",
        ):
            logging.info(f"{corpus_name} subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"{corpus_name}_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"{corpus_name}_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
