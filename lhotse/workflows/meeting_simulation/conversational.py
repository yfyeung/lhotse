import functools
import itertools
import logging
import random
from collections import defaultdict
from typing import Any, List, Optional, Union

import numpy as np

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet, MixedCut, MixTrack, MonoCut
from lhotse.cut.set import mix
from lhotse.utils import uuid4
from lhotse.workflows.meeting_simulation.base import (
    BaseMeetingSimulator,
    reverberate_cuts,
)


class ConversationalMeetingSimulator(BaseMeetingSimulator):
    """
    This simulator uses the method described in https://arxiv.org/abs/2204.00890 and
    implemented in https://github.com/BUTSpeechFIT/EEND_dataprep. Note that a similar
    method of meeting simulation is also implemented in
    https://github.com/jsalt2020-asrdiar/jsalt2020_simulate.

    The basic idea is to sample the silence and overlap durations collectively for all
    speakers so that we get similar speech/silence/overlap characteristics as the
    original distribution. In general, this method produces more realistic overlap
    durations than the `SpeakerIndependentMeetingSimulator`. This is done by learning
    the histogram of 3 distributions: same speaker pause, different speaker pause, and
    different speaker overlap. In this implementation, we learn the histograms from
    provided data, otherwise we use the initialization values as shift of a Gamma
    distribution with scale 1.0.
    """

    def __init__(
        self,
        same_spk_pause: float = 2.0,
        diff_spk_pause: float = 3.0,
        diff_spk_overlap: float = 2.0,
    ):
        """
        :param same_spk_pause: the mean pause duration between utterances of the same
            speaker. [Default: 2.0]
        :param diff_spk_pause: the mean pause duration between utterances of different
            speakers. [Default: 3.0]
        :param diff_spk_overlap: the mean overlap duration between utterances of
            different speakers. [Default: 2.0]
        """
        super().__init__()
        for duration in [same_spk_pause, diff_spk_pause, diff_spk_overlap]:
            assert duration is None or duration > 0, "Durations must be > 0."

        self.same_spk_pause_dist = same_spk_pause
        self.diff_spk_pause = diff_spk_pause
        self.diff_spk_overlap = diff_spk_overlap

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} "
            f"(same_spk_pause={self.same_spk_pause}, "
            f"diff_spk_pause={self.diff_spk_pause}, "
            f"diff_spk_overlap={self.diff_spk_overlap})"
        )

    def _init_defaults(self):
        from scipy.stats import gamma

        self.same_spk_pause_dist = gamma(a=1.0, scale=1.0, loc=self.same_spk_pause)
        self.diff_spk_pause_dist = gamma(a=1.0, scale=1.0, loc=self.diff_spk_pause)
        self.diff_spk_overlap_dist = gamma(a=1.0, scale=1.0, loc=self.diff_spk_overlap)

    def _compute_histogram_dist(self, values: np.ndarray) -> Any:
        """
        Compute the histogram of the given values and return the bin edges and the
        corresponding probabilities.
        """
        from scipy.stats import rv_histogram

        hist, bin_edges = np.histogram(values, bins=100, density=True)
        return rv_histogram((hist, bin_edges))

    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        :param meetings: a SupervisionSet containing the meetings to be used for
        """
        if meetings is None:
            logging.info(f"No meetings provided, using default parameters.")
            self._init_defaults()
            return

        assert isinstance(
            meetings, SupervisionSet
        ), "The meetings must be provided as a SupervisionSet."

        from cytoolz.itertoolz import groupby

        # Generate same speaker pause distribution.
        same_spk_pause_values = []

        speaker_segments = groupby(
            lambda s: (s.recording_id, s.speaker),
            sorted(meetings, key=lambda s: (s.recording_id, s.speaker)),
        )

        for segments in speaker_segments.values():
            segments = sorted(segments, key=lambda s: s.start)
            for i in range(1, len(segments)):
                same_spk_pause_values.append(
                    max(0, segments[i].start - segments[i - 1].end)
                )

        self.same_spk_pause_dist = self._compute_histogram_dist(
            np.array(same_spk_pause_values)
        )

        # Generate different speaker pause and overlap distributions.
        diff_spk_pause_values = []
        diff_spk_overlap_values = []

        recording_segments = groupby(
            lambda s: s.recording_id,
            sorted(meetings, key=lambda s: s.recording_id),
        )
        for segments in recording_segments.values():
            segments = sorted(segments, key=lambda s: s.start)
            for i in range(1, len(segments)):
                if segments[i].speaker == segments[i - 1].speaker:
                    continue
                if segments[i].start > segments[i - 1].end:
                    diff_spk_pause_values.append(
                        segments[i].start - segments[i - 1].end
                    )
                else:
                    diff_spk_overlap_values.append(
                        segments[i - 1].end - segments[i].start
                    )

        self.diff_spk_pause_dist = self._compute_histogram_dist(
            np.array(diff_spk_pause_values)
        )
        self.diff_spk_overlap_dist = self._compute_histogram_dist(
            np.array(diff_spk_overlap_values)
        )

        print(f"Learned parameters: {self}")

    @functools.lru_cache(maxsize=128)
    def _get_speaker_cuts(self, base_cuts_idx: int, speaker: str) -> CutSet:
        """
        Return all cuts for the given speaker.
        """
        return list(
            self.cuts[base_cuts_idx]
            .filter(lambda cut: cut.supervisions[0].speaker == speaker)
            .cuts.values()
        )

    def _create_mixture(self, utterances: List[MonoCut]) -> MixedCut:
        """
        Create a MixedCut object from a list of MonoCuts (utterances).
        We sample pauses and/or overlaps from the initilized or learned distributions.
        Then, we create a MixedCut where each track represents a different speaker.
        """
        # First sample offsets for each utterance.
        offsets = []
        for i in range(1, len(utterances) + 1):
            if (
                utterances[i].supervisions[0].speaker
                == utterances[i - 1].supervisions[0].speaker
            ):
                offsets.append(self.same_spk_pause_dist.rvs())
            else:
                if self.bernoulli.rvs():
                    offsets.append(self.diff_spk_pause_dist.rvs())
                else:
                    offsets.append(-self.diff_spk_overlap_dist.rvs())

        # Group utterances by speaker and compute net offset (i.e. offset w.r.t previous
        # utterance of the same speaker).
        spk_tracks = defaultdict(list)
        # Add first cut to the dictionary. Each dictionary element contains a list of
        # tracks belonging to a specific speaker. List elements are tuples of the form
        # (cut, start_time).
        spk_tracks[utterances[0].supervisions[0].speaker].append(
            (
                utterances[0],
                0.0,
            )
        )
        cur_end = utterances[0].duration

        # Iterate over the rest of the cuts
        for offset, utt in zip(offsets, utterances[1:]):
            spk = utt.supervisions[0].speaker
            cur_start = max(0, cur_end + offset)
            spk_tracks[spk].append((utt, cur_start))
            cur_end = max(cur_start + utt.duration, cur_end)

        # Now create speaker-wise tracks by mixing their utterances with silence padding.
        tracks = []
        for spk, spk_utts in spk_tracks.items():
            track, start = spk_utts[0]
            for utt, offset in spk_utts[1:]:
                track = mix(track, utt, offset=offset, allow_padding=True)
            track = MixTrack(cut=track, type=type(track), offset=start)
            tracks.append(track)

        return MixedCut(id=str(uuid4()), tracks=tracks)

    def simulate(
        self,
        *cuts: CutSet,
        num_meetings: Optional[int] = None,
        num_repeats: Optional[int] = None,
        num_speakers_per_meeting: Union[int, List[int]] = 2,
        speaker_count_probs: Optional[List[float]] = None,
        min_utts_per_speaker: int = 5,
        max_utts_per_speaker: int = 10,
        seed: int = 0,
    ) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        :param cuts: one or more CutSet containing the MonoCut objects to be used for simulation.
            If multiple CutSets are provided, each mixture will contain cuts sampled from
            a specific CutSet. This may be useful when we want to simulate mixtures with
            cuts from the same recording.
        :param num_meetings: the number of meetings to simulate.
        :param num_repeats: the number of times to repeat the provided cuts. This means that
            the number of simulated meetings depends on how many cuts are available.
        :param num_speakers_per_meeting: the number of speakers per meeting. If a list is
            provided, the number of speakers per meeting is sampled from this list.
            [Default: 2]
        :param speaker_count_probs: the probability of each number of speakers per meeting.
            [Default: None]
        :param min_utts_per_speaker: the minimum number of utterances per speaker to be
            used for simulation. [Default: 5]
        :param max_utts_per_speaker: the maximum number of utterances per speaker to be
            used for simulation. [Default: 10]
        :param seed: the random seed to be used for simulation. [Default: 0]
        """
        assert len(cuts) > 0, "At least one CutSet must be provided."

        if num_meetings is None and num_repeats is None:
            raise ValueError("Either num_meetings or num_repeats must be provided.")

        if isinstance(num_speakers_per_meeting, int):
            num_speakers_per_meeting = [num_speakers_per_meeting]

        if speaker_count_probs is None:
            speaker_count_probs = [1.0 / len(num_speakers_per_meeting)] * len(
                num_speakers_per_meeting
            )

        assert len(num_speakers_per_meeting) == len(
            speaker_count_probs
        ), "The number of speakers per meeting and the number of probabilities must be the same."

        self.cuts = cuts
        # Make sure there are only MonoCuts in the CutSets.
        assert all(
            len(base_cuts.simple_cuts) == len(base_cuts) for base_cuts in cuts
        ), "The CutSets must contain only MonoCuts. "

        # Create random number generators with the given seed.
        npr = np.random.RandomState(seed)
        rand = random.Random(seed)

        # Reset speaker bucket cache.
        self._get_speaker_cuts.cache_clear()

        self.bernoulli = npr.binomial(1, 0.5)

        mixtures = []

        for _ in range(num_meetings):

            # Sample the cut-set that this meeting will be generated from.
            base_cuts_idx = npr.randint(len(cuts))
            base_cuts = cuts[base_cuts_idx]

            # Sample the number of speakers for this meeting.
            num_speakers = npr.choice(num_speakers_per_meeting, p=speaker_count_probs)

            # Sample the speakers for this meeting.
            speakers = rand.sample(
                base_cuts.speakers,
                k=min(num_speakers, len(base_cuts.speakers)),
            )

            # Sample the utterances for each speaker. We use `replace=True` here because
            # we may need to use the same utterance multiple times to get the desired
            # number of utterances per speaker.
            utterances = [
                rand.choices(
                    self._get_speaker_cuts(base_cuts_idx, speaker),
                    k=npr.randint(min_utts_per_speaker, max_utts_per_speaker + 1),
                )
                for speaker in speakers
            ]

            # Flatten the list of lists and randomly permute the utterances.
            utterances = list(itertools.chain(*utterances))
            rand.shuffle(utterances)

            # Create the meeting.
            mixture = self._create_mixture(utterances)

            mixtures.append(mixture)

        return CutSet.from_cuts(mixtures)

    def reverberate(self, cuts: CutSet, *rirs: RecordingSet) -> CutSet:
        return reverberate_cuts(cuts, *rirs)
