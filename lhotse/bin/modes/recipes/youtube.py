from typing import Dict, List, Optional, Tuple, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.youtube import prepare_youtube
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_name", type=str)
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--language", type=str, help="Language of the corpus.")
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def youtube(
    corpus_name: str,
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: Optional[str] = None,
    num_jobs: int = 1,
):
    """YouTube data preparation."""
    prepare_youtube(
        corpus_name=corpus_name,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        language=language,
        num_jobs=num_jobs,
    )
