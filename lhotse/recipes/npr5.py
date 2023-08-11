import logging
import os
import json
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import set_ffmpeg_torchaudio_info_enabled
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


NPR5 = ("all_things_2005", "all_things_2006", "all_things_2007", "all_things_2008", "all_things_2009", "all_things_2010", "all_things_2011", "all_things_2012", "all_things_2013", "all_things_2014", "all_things_2015", "all_things_2016", "all_things_2017", "all_things_2018", "all_things_2019", "all_things_2020", "all_things_2021", "all_things_2022", "all_things_2023")


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    file_name = str(audio_path).replace(".mp3", "").replace(str(corpus_dir) + "/", "")
    text_path = os.path.dirname(audio_path) + "/" + "t_edited.txt" 
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    with open(text_path) as f:
        text = f.read()

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
    part_path = corpus_dir / subset / "scripts_step2" / "data"
    audio_paths = list(part_path.rglob("*.mp3"))

    with ProcessPoolExecutor(num_jobs) as ex:
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


def prepare_npr5(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the NPR5 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir) if output_dir is not None else None

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing NPR5...")

    set_ffmpeg_torchaudio_info_enabled(False)

    subsets = NPR5

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing NPR5 subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="npr5",
            suffix="jsonl.gz",
        ):
            logging.info(f"NPR5 subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"npr5_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"npr5_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    set_ffmpeg_torchaudio_info_enabled(True)

    return manifests


if __name__ == "__main__":
    prepare_npr5("/ceph-data3/xiaoyu/npr5", ".", num_jobs=32)
