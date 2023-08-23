import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

AISHELL = (
    "train",
    "dev",
    "test",
)


def text_normalize(line: str):
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/aishell_data_prep.sh#L54
    sed 's/ａ/a/g' | sed 's/ｂ/b/g' |\
    sed 's/ｃ/c/g' | sed 's/ｋ/k/g' |\
    sed 's/ｔ/t/g' > $dir/transcripts.t

    """
    line = line.replace("ａ", "a")
    line = line.replace("ｂ", "b")
    line = line.replace("ｃ", "c")
    line = line.replace("ｋ", "k")
    line = line.replace("ｔ", "t")
    line = line.upper()
    return line


def prepare_aishell(
    corpus_dir: Pathlike,
    discrete_token_path: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    discrete_token_path = Path(discrete_token_path)
    assert discrete_token_path.is_file(), f"No such file: {discrete_token_path}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = corpus_dir / "transcript/aishell_transcript_v0.8.txt"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            idx_transcript = line.split()
            content = " ".join(idx_transcript[1:])
            content = text_normalize(content)
            transcript_dict[idx_transcript[0]] = content

    dataset_parts = AISHELL

    manifests = defaultdict(dict)

    discrete_tokens_info = {}
    with open(discrete_token_path) as f:
        discrete_tokens = f.read().splitlines()
        for discrete_token in discrete_tokens:
            discrete_token = discrete_token.split(" ", 1)
            discrete_tokens_info[discrete_token[0]] = discrete_token[1]

    for part in tqdm(
        dataset_parts,
        desc="Process aishell audio, it takes about 102 seconds.",
    ):
        logging.info(f"Processing aishell subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "wav" / f"{part}"
        for audio_path in wav_path.rglob("**/*.wav"):
            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            discrete_tokens = discrete_tokens_info[idx]
            recording = Recording.from_file(audio_path)
            recordings.append(recording)
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="Chinese",
                speaker=speaker,
                text=text.strip(),
            )
            segment.discrete_tokens = discrete_tokens
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"aishell_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"aishell_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests


if __name__ == "__main__":
    prepare_aishell("/star-data/kangwei/data/data_aishell", "/k2-dev/yangyifan/icefall-disc/egs/aishell/wavlm_large_l24_kms2000/download/DiscreteAudioToken/wavlm_large_l24_kms2000/out_quantized", ".")
