import logging
import os
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

DISC_TTS = ("dac", "encodec", "gt", "hifigan", "hubert", "vq", "wavlm")


def _parse_utterance(
    corpus_dir: Pathlike,
    audio_path: Pathlike,
    audio_id: str,
    text: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=audio_id,
    )
    segment = SupervisionSegment(
        id=audio_id,
        recording_id=audio_id,
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
    corpus_dir = Path(corpus_dir)
    text_path = corpus_dir / "text"
    part_path = corpus_dir / "audios"/ subset
    audio_paths = list(part_path.rglob("*.wav"))

    with open(text_path) as f:
        texts = f.read().splitlines()

    text_info = {}
    for text in texts:
        text = text.split(" ", 1)
        text_info[text[0]] = text[1]

    with ThreadPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_path in tqdm(audio_paths, desc="Distributing tasks"):
            audio_id = (
                str(audio_path).replace(".wav", "").replace(str(part_path) + "/", "")
            )
            text = text_info[audio_id]
            futures.append(
                ex.submit(_parse_utterance, corpus_dir, audio_path, audio_id, text)
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


def prepare_disc_tts(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the DiscTTS dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing DiscTTS...")

    subsets = DISC_TTS

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing DiscTTS subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="disc_tts",
            suffix="jsonl.gz",
        ):
            logging.info(f"DiscTTS subset: {part} already prepared - skipping.")
            continue

        recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"disc_tts_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"disc_tts_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_disc_tts(
        "/k2-dev/yangyifan/icefall-disc/egs/disc_tts/ASR/download/disc-tts",
        ".",
        num_jobs=16,
    )
