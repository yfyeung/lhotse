import logging
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract

LIBRISPEECH = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


def prepare_librispeech(
    corpus_dir: Pathlike,
    discrete_token_path: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    discrete_token_path = Path(discrete_token_path)
    assert discrete_token_path.is_file(), f"No such file: {discrete_token_path}"

    if dataset_parts == "auto":
        dataset_parts = LIBRISPEECH
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    discrete_tokens_info = {}
    with open(discrete_token_path) as f:
        discrete_tokens = f.read().splitlines()
        for discrete_token in discrete_tokens:
            discrete_token = discrete_token.split(" ", 1)
            discrete_tokens_info[discrete_token[0]] = discrete_token[1]

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing LibriSpeech subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir):
                logging.info(f"LibriSpeech subset: {part} already prepared - skipping.")
                continue
            recordings = []
            supervisions = []
            part_path = corpus_dir / part
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
                        futures.append(
                            ex.submit(parse_utterance, part_path, line, discrete_tokens_info)
                        )

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"librispeech_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"librispeech_recordings_{part}.jsonl.gz"
                )

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests


def parse_utterance(
    dataset_split_path: Path,
    line: str,
    discrete_tokens_info: dict,
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
    discrete_tokens = discrete_tokens_info[recording_id]
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
    segment.discrete_tokens = discrete_tokens

    return recording, segment


if __name__ == "__main__":
    prepare_librispeech("/star-data/yifan/LibriSpeech", "/k2-dev/yangyifan/icefall-disc/egs/librispeech/encodec/download/DiscreteAudioToken/encodec/out_quantized", "auto", ".", num_jobs=16)
