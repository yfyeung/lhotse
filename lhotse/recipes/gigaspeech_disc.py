import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    compute_num_samples,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available

GIGASPEECH_PARTS = ("XL", "L", "M", "S", "XS", "DEV", "TEST")


def prepare_gigaspeech(
    corpus_dir: Pathlike,
    discrete_token_path: Pathlike,
    output_dir: Optional[Pathlike],
    dataset_parts: Union[str, Sequence[str]] = "auto",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    if is_module_available("speechcolab"):
        from speechcolab.datasets.gigaspeech import GigaSpeech
    else:
        raise ImportError(
            "To process the GigaSpeech corpus, please install optional dependency: pip install speechcolab"
        )

    subsets = ("XL",) if dataset_parts == "auto" else dataset_parts
    if isinstance(subsets, str):
        subsets = [subsets]
    corpus_dir = Path(corpus_dir)
    gigaspeech = GigaSpeech(corpus_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Maybe some manifests already exist: we can read them and save a bit of preparation time.
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=output_dir,
        prefix="gigaspeech",
        suffix="jsonl.gz",
        lazy=True,
    )

    discrete_tokens_info = {}
    with open(discrete_token_path) as f:
        discrete_tokens = f.read().splitlines()
        for discrete_token in discrete_tokens:
            discrete_token = discrete_token.split(" ", 1)
            discrete_tokens_info[discrete_token[0]] = discrete_token[1]

    for part in subsets:
        logging.info(f"Processing GigaSpeech subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="gigaspeech", suffix="jsonl.gz"
        ):
            logging.info(f"GigaSpeech subset: {part} already prepared - skipping.")
            continue

        with RecordingSet.open_writer(
            output_dir / f"gigaspeech_recordings_{part}.jsonl.gz"
        ) as rec_writer, SupervisionSet.open_writer(
            output_dir / f"gigaspeech_supervisions_{part}.jsonl.gz"
        ) as sup_writer:
            for recording, segments in tqdm(
                parallel_map(
                    parse_utterance,
                    gigaspeech.audios("{" + part + "}"),
                    repeat(gigaspeech.gigaspeech_dataset_dir),
                    repeat(discrete_tokens_info),
                    num_jobs=num_jobs,
                ),
                desc="Processing GigaSpeech JSON entries",
            ):
                # Fix and validate the recording + supervisions
                recordings, segments = fix_manifests(
                    recordings=RecordingSet.from_recordings([recording]),
                    supervisions=SupervisionSet.from_segments(segments),
                )
                validate_recordings_and_supervisions(
                    recordings=recordings, supervisions=segments
                )

                # Write the manifests
                rec_writer.write(recordings[0])
                for s in segments:
                    sup_writer.write(s)

        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
            "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
        }

    return dict(manifests)


def parse_utterance(
    audio: Any,
    root_path: Path,
    discrete_tokens_info: dict,
) -> Optional[Tuple[Recording, List[SupervisionSegment]]]:
    sampling_rate = int(audio["sample_rate"])
    recording = Recording(
        id=audio["aid"],
        sources=[
            AudioSource(
                type="file",
                channels=list(range(int(audio["channels"]))),
                source=str(root_path / audio["path"]),
            )
        ],
        num_samples=compute_num_samples(
            duration=Seconds(audio["duration"]), sampling_rate=sampling_rate
        ),
        sampling_rate=sampling_rate,
        duration=Seconds(audio["duration"]),
    )
    segments = []
    for seg in audio["segments"]:
        segment = SupervisionSegment(
            id=seg["sid"],
            recording_id=audio["aid"],
            start=Seconds(seg["begin_time"]),
            duration=round(Seconds(seg["end_time"] - seg["begin_time"]), ndigits=8),
            channel=0,
            language="English",
            speaker=seg["speaker"],
            text=seg["text_tn"],
        )
        try:
            segment.discrete_tokens = discrete_tokens_info[seg["sid"]]
        except:
            logging.warning(f"No discrete tokens for segment: {seg['sid']}")
            continue

        segments.append(segment)
    return recording, segments


if __name__ == "__main__":
    prepare_gigaspeech(
        "/mnt/lustre/sjtu/shared/data/asr/rawdata/GigaSpeech",
        "/mnt/lustre/sjtu/home/fys18/code/DiscreteAudioToken/DataProcess/data/gigaspeech-xl/wavlm_large_l21_kms2000_sp1.0_nopad/out_quantized",
        ".",
        "auto",
        num_jobs=16,
    )
