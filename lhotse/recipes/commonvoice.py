"""
Official description from the "about" page of the Mozilla CommonVoice project
(source link: https://commonvoice.mozilla.org/en/about)

Why Common Voice?
Mozilla Common Voice is an initiative to help teach machines how real people speak.
This project is an effort to bridge the digital speech divide. Voice recognition technologies bring a human dimension to our devices, but developers need an enormous amount of voice data to build them. Currently, most of that data is expensive and proprietary. We want to make voice data freely and publicly available, and make sure the data represents the diversity of real people. Together we can make voice recognition better for everyone.

How does it work?
We are crowdsourcing an open-source dataset of voices. Donate your voice, validate the accuracy of other people's clips, make the dataset better for everyone.
"""
import csv
import logging
import math
import numbers
import shutil
import tarfile
import warnings
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context as mp_get_context
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    get_ffmpeg_torchaudio_info_enabled,
    load_manifest,
    set_ffmpeg_torchaudio_info_enabled,
    validate_recordings_and_supervisions,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract

COMMONVOICE = (
    "https://huggingface.co/mozilla-foundation"
)


COMMONVOICE_LANGS = "en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk".split()
COMMONVOICE_SPLITS = ("train", "dev", "test", "validated", "invalidated", "other")
COMMONVOICE_DEFAULT_SPLITS = ("test", "train")


@contextmanager
def disable_ffmpeg_torchaudio_info() -> None:
    enabled = get_ffmpeg_torchaudio_info_enabled()
    set_ffmpeg_torchaudio_info_enabled(False)
    try:
        yield
    finally:
        set_ffmpeg_torchaudio_info_enabled(enabled)


def _read_cv_manifests_if_cached(
    output_dir: Optional[Pathlike],
    language: str,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns:
        {'train': {'recordings': ..., 'supervisions': ...}, 'dev': ..., 'test': ...}
    """
    if output_dir is None:
        return {}
    manifests = defaultdict(dict)
    for part in ["train", "dev", "test"]:
        for manifest in ["recordings", "supervisions"]:
            path = output_dir / f"cv_{manifest}_{language}_{part}.jsonl.gz"
            if not path.is_file():
                continue
            manifests[part][manifest] = load_manifest(path)
    return manifests


def _parse_utterance(
    lang_path: Path,
    language: str,
    part: str,
    audio_info: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    audio_path = lang_path / part / audio_info["path"]

    if not audio_path.is_file():
        logging.info(f"No such file: {audio_path}")
        return None

    recording_id = Path(audio_info["path"]).stem
    recording = Recording.from_file(path=audio_path, recording_id=recording_id)

    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
        text=audio_info["sentence"].strip(),
    )
    return recording, segment


def _prepare_part(
    lang: str,
    part: str,
    lang_path: Pathlike,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Prepares part of CommonVoice data.

    :param lang: string language code (e.g., "en").
    :param part: which split to prepare (e.g., "train", "validated", etc.).
    :param lang_path: path to a CommonVoice directory for a specific language
        (e.g., "/path/to/cv-corpus-13.0-2023-03-09/pl").
    :param num_jobs: How many concurrent workers to use for scanning of the audio files.
    :return: a tuple of (RecordingSet, SupervisionSet) objects,
        note that CommonVoice manifests may be fairly large in memory.
    """

    lang_path = Path(lang_path)
    tsv_path = lang_path / f"{part}.tsv"

    with disable_ffmpeg_torchaudio_info():
        with ProcessPoolExecutor(
            max_workers=num_jobs,
            mp_context=mp_get_context("spawn"),
        ) as ex:

            futures = []
            recordings = []
            supervisions = []
            audio_infos = []

            with open(tsv_path, "r") as f:

                # Note: using QUOTE_NONE as CV dataset contains unbalanced quotes, cleanup needed later
                audio_infos = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)

                for audio_info in tqdm(audio_infos, desc="Distributing tasks"):
                    futures.append(
                        ex.submit(
                            _parse_utterance,
                            lang_path,
                            lang,
                            part,
                            audio_info,
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


def prepare_commonvoice(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    languages: Union[str, Sequence[str]] = "auto",
    splits: Union[str, Sequence[str]] = COMMONVOICE_DEFAULT_SPLITS,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    This function expects the input directory structure of::

        >>> metadata_path = corpus_dir / language_code / "{train,dev,test}.tsv"
        >>> # e.g. pl_train_metadata_path = "/path/to/cv-corpus-13.0-2023-03-09/pl/train.tsv"
        >>> audio_path = corpus_dir / language_code / part
        >>> # e.g. pl_audio_path = "/path/to/cv-corpus-13.0-2023-03-09/pl/part"

    Returns a dict with 3-level structure (lang -> split -> manifest-type)::

        >>> {'en/fr/pl/...': {'train/dev/test': {'recordings/supervisions': manifest}}}

    :param corpus_dir: Pathlike, the path to the downloaded corpus.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param languages: 'auto' (prepare all discovered data) or a list of language codes.
    :param splits: by default ``['train', 'dev', 'test']``, can also include
        ``'validated'``, ``'invalidated'``, and ``'other'``.
    :param num_jobs: How many concurrent workers to use for scanning of the audio files.
    :return: a dict with manifests for all specified languagues and their train/dev/test splits.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert output_dir is not None, (
        "CommonVoice recipe requires to specify the output "
        "manifest directory (output_dir cannot be None)."
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if languages == "auto":
        languages = set(COMMONVOICE_LANGS).intersection(
            path.name for path in corpus_dir.glob("*")
        )
        if not languages:
            raise ValueError(
                f"Could not find any of CommonVoice languages in: {corpus_dir}"
            )
    elif isinstance(languages, str):
        languages = [languages]

    manifests = {}

    for lang in tqdm(languages, desc="Processing CommonVoice languages"):
        logging.info(f"Language: {lang}")
        lang_path = corpus_dir / lang

        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        # Pattern: "cv_recordings_en_train.jsonl.gz" / "cv_supervisions_en_train.jsonl.gz"
        lang_manifests = _read_cv_manifests_if_cached(
            output_dir=output_dir, language=lang
        )

        for part in tqdm(splits, desc=f"Spliting"):
            logging.info(f"Spliting {part}")
            if part in lang_manifests:
                logging.info(
                    f"{part} split of CommonVoice-{lang} already prepared - skipping."
                )
                continue
            recording_set, supervision_set = _prepare_part(
                lang=lang,
                part=part,
                lang_path=lang_path,
                num_jobs=num_jobs,
            )

            # Fix manifests
            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            supervision_set.to_file(
                output_dir / f"cv-{lang}_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"cv-{lang}_recordings_{part}.jsonl.gz")

            lang_manifests[part] = {
                "supervisions": supervision_set,
                "recordings": recording_set,
            }

        manifests[lang] = lang_manifests

    return manifests


if __name__ == "__main__":
    prepare_commonvoice(
        "/scratch/icefall/egs/commonvoice/ASR/download/commonvoice_17.0",
        "/scratch/icefall/egs/commonvoice/ASR/data/manifests",
        "th",
        num_jobs=1,
    )