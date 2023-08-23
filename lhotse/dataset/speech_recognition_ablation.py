import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader, default_collate

from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_custom_field
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix


class K2SpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task using k2 library.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - single-channel:
                        - features: (B, T, F)
                        - audio: (B, T)
                      - multi-channel: currently not supported
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S

                    # For feature input strategies
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)

                    # For audio input strategies
                    'start_sample': Tensor[int] of shape (S,)
                    'num_samples': Tensor[int] of shape (S,)

                    # Optionally, when return_cuts=True
                    'cut': List[AnyCut] of len S
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        """
        k2 ASR IterableDataset constructor.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_frames and max_cuts.
        """
        validate_for_asr(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        has_word_alignments = all(
            s.alignment is not None and "word" in s.alignment
            for c in cuts
            for s in c.supervisions
        )
        if has_word_alignments:
            # TODO: might need to refactor BatchIO API to move the following conditional logic
            #       into these objects (e.g. use like: self.input_strategy.convert_timestamp(),
            #       that returns either num_frames or num_samples depending on the strategy).
            words, starts, ends = [], [], []
            frame_shift = cuts[0].frame_shift
            sampling_rate = cuts[0].sampling_rate
            if frame_shift is None:
                try:
                    frame_shift = self.input_strategy.extractor.frame_shift
                except AttributeError:
                    raise ValueError(
                        "Can't determine the frame_shift -- it is not present either in cuts or the input_strategy. "
                    )
            for c in cuts:
                for s in c.supervisions:
                    words.append([aliword.symbol for aliword in s.alignment["word"]])
                    starts.append(
                        [
                            compute_num_frames(
                                aliword.start,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
                    ends.append(
                        [
                            compute_num_frames(
                                aliword.end,
                                frame_shift=frame_shift,
                                sampling_rate=sampling_rate,
                            )
                            for aliword in s.alignment["word"]
                        ]
                    )
            batch["supervisions"]["word"] = words
            batch["supervisions"]["word_start"] = starts
            batch["supervisions"]["word_end"] = ends

        return batch


class DiscretizedInputSpeechRecognitionDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the speech recognition task that provides discrete audio tokens instead of audios/features.
    In this implementation, there will always be a single channel.

    Returns:

    .. code-block::

        {
            'token': (B x Tokens) int tensor
            'token_lens': (B, ) int tensor
        }
    """

    def __init__(
        self,
        field: str,
        num_tokens: int,
        token_type: str,
        frequency_size: Optional[int] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.field = field
        self.num_tokens = num_tokens
        self.frequency_size = frequency_size
        self.token_type = token_type
        self.input_transforms = ifnone(input_transforms, [])

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        if self.token_type in ("wavlm", "hubert"):
            tokens = []
            token_lens = []
            for c in cuts:
                token = torch.tensor(
                    list(map(int, c.discrete_tokens.split())), dtype=torch.int64
                )
                tokens.append(token)
                token_lens.append(token.size(0))
            tokens = pad_sequence(
                tokens, batch_first=True, padding_value=self.num_tokens
            )
            token_lens = torch.tensor(token_lens, dtype=torch.int64)
        elif self.token_type == "vq-wav2vec":
            tokens = []
            token_lens = []
            for c in cuts:
                token = torch.tensor(
                    list(map(int, c.discrete_tokens.split())), dtype=torch.int64
                )
                token_len = len(token) >> 1
                tokens.append(token.reshape(2, token_len).T)
                token_lens.append(token_len)
            tokens = pad_sequence(
                tokens, batch_first=True, padding_value=self.num_tokens
            )
            token_lens = torch.tensor(token_lens, dtype=torch.int64)
        elif self.token_type == "encodec":
            tokens = []
            token_lens = []
            for c in cuts:
                token = torch.tensor(
                    list(map(int, c.discrete_tokens.split())), dtype=torch.int64
                )
                token_len = len(token) >> 3
                tokens.append(token.reshape(8, token_len).T)
                token_lens.append(token_len)
            tokens = pad_sequence(
                tokens, batch_first=True, padding_value=self.num_tokens
            )
            token_lens = torch.tensor(token_lens, dtype=torch.int64)

        data_dict = {}
        for tnfm in self.input_transforms:
            if tnfm.__class__.__name__ == "DiscretizedInputAugment":
                tokens, frequency_masks = tnfm(
                    tokens, self.num_tokens, self.frequency_size
                )
                data_dict["frequency_masks"] = frequency_masks
            else:
                tokens = tnfm(tokens)

        data_dict["cuts"] = cuts
        data_dict["tokens"] = tokens
        data_dict["token_lens"] = token_lens

        return data_dict

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)
        assert all(cut.has_recording for cut in cuts)


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )

            # Supervision start time is relative to Cut ...
            # https://lhotse.readthedocs.io/en/v0.10_e/cuts.html
            #
            # 'supervision.end' is end of supervision inside the Cut
            assert supervision.end <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )
