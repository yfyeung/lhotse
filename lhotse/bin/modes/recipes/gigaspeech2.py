from typing import Dict, List, Optional, Tuple, Union

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.gigaspeech2 import prepare_gigaspeech2
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--lang", type=str)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def gigaspeech2(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    lang: Optional[str] = None,
    num_jobs: int = 1,
):
    """GigaSpeech2 data preparation."""
    prepare_gigaspeech2(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        lang=lang,
        num_jobs=num_jobs,
    )
