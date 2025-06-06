import logging
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import textgrid
from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, add_durations


def _parse_textgrid(
    corpus_dir: Pathlike,
    textgrid_path: Pathlike,
    language: str = None,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    def default_filter(interval):
        return len(interval.mark) > 0 and interval.mark != "MSA Fusha not dialect"

    file_name = textgrid_path.stem
    audio_path = corpus_dir / (file_name + ".wav")
    audio_path = audio_path.resolve()
    assert audio_path.is_file(), f"No such file: {audio_path}"

    tg = textgrid.TextGrid.fromFile(textgrid_path)
    filtered_intervals = (i for i in tg.tiers[0].intervals if default_filter(i))

    recording = Recording.from_file(
        path=audio_path,
        recording_id=file_name,
    )

    supervisions = []
    for seq, interval in enumerate(filtered_intervals):
        supervision = SupervisionSegment(
            id=f"{file_name}-{seq}",
            recording_id=file_name,
            start=interval.minTime,
            duration=add_durations(
                interval.maxTime, -interval.minTime, sampling_rate=16000
            ),
            channel=0,
            text=interval.mark.strip(),
            language=language,
        )
        supervisions.append(supervision)

    return recording, supervisions


def _prepare(
    corpus_dir: Pathlike,
    textgrid_dir: Pathlike,
    language: str = None,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param corpus_dir: Path to the dataset.
    :param textgrid_dir: Path to the textgrid directory.
    :param language: str of the language of corpus.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    textgrid_paths = list(textgrid_dir.rglob("*.TextGrid"))

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for textgrid_path in tqdm(textgrid_paths, desc="Distributing tasks"):
            futures.append(
                ex.submit(_parse_textgrid, corpus_dir, textgrid_path, language)
            )

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recordings.append(result[0])
            supervisions.extend(result[1])

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_from_textgrids(
    corpus_name: str,
    corpus_dir: Pathlike,
    textgrid_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: Optional[str] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_name: str of the corpus name.
    :param corpus_dir: Path to the dataset.
    :param textgrid_dir: Path to the textgrid directory.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param language: str of the language of corpus.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    textgrid_dir = Path(textgrid_dir)
    assert textgrid_dir.is_dir(), f"No such directory: {textgrid_dir}"

    logging.info("Preparing from textgrid files...")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    part = textgrid_dir.stem
    logging.info(f"Processing textgrid directory: {textgrid_dir}")
    if manifests_exist(
        part=part,
        output_dir=output_dir,
        prefix=corpus_name,
        suffix="jsonl.gz",
    ):
        logging.info(f"textgrid directory: {textgrid_dir} already prepared - skipping.")
        return manifests

    recording_set, supervision_set = _prepare(
        corpus_dir, textgrid_dir, language, num_jobs
    )

    if output_dir is not None:
        supervision_set.to_file(
            output_dir / f"{corpus_name}_supervisions_{part}.jsonl.gz"
        )
        recording_set.to_file(output_dir / f"{corpus_name}_recordings_{part}.jsonl.gz")

    manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
