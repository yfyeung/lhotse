from typing import Dict, List, Optional, Tuple, Union

import click

from lhotse.bin.modes import prepare
from lhotse.recipes.textgrids import prepare_from_textgrids
from lhotse.utils import Pathlike


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_name", type=str)
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("textgrid_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option("--language", type=str, help="Language of the corpus.")
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def textgrids(
    corpus_name: str,
    corpus_dir: Pathlike,
    textgrid_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    language: Optional[str] = None,
    num_jobs: int = 1,
):
    """TextGrid files preparation."""
    prepare_from_textgrids(
        corpus_name=corpus_name,
        corpus_dir=corpus_dir,
        textgrid_dir=textgrid_dir,
        output_dir=output_dir,
        language=language,
        num_jobs=num_jobs,
    )
