# ANNI: Auditory Nerve Noticeability Index

ANNI is a full-reference measure of how noticeable the difference between a reference and a test audio signal is.
It simulates how the auditory nerve responds to each signal and compares the two responses the way an ideal observer
would. The score is in d′ units, the standard unit of detectability: **0 means no difference, and larger values mean
a more noticeable difference.** ANNI is built for small differences near the threshold of hearing, such as audio
watermarks.

> **Status: research prototype (v0.1).** ANNI is calibrated so that D = 1 marks a just-noticeable difference: the
> point at which listeners in classic laboratory experiments start to detect a change. A score below 1 should
> therefore mean the difference is hard to hear, even when comparing directly. This has not yet been confirmed by
> listening tests on real-world audio, so we can't guarantee it.

## Install

```bash
pip install anni
```

Requires Python 3.9 or newer, PyTorch, NumPy and soxr.

For development, install from a clone in editable mode:

```bash
git clone https://github.com/deepmark/anni.git
cd anni
python -m pip install -e ".[test]"
```

## Usage

```python
import soundfile as sf          # any audio loader works
from anni import ANNI

reference, sr = sf.read("original.wav")
test, _ = sf.read("watermarked.wav")

anni = ANNI()
anni(reference, test, sr=sr)
# e.g. {'D_peak': 1.49, 'D_med': 0.81}
```

| Score | Meaning |
|---|---|
| `D_peak` | how noticeable the difference is at the most noticeable 200 ms moment |
| `D_med` | how noticeable it is at a typical moment (median over 200 ms windows) |

Lower is better. To compare two processed versions, score both against the same reference. A difference between two
scores can be expressed in dB as 20·log₁₀(D_B / D_A).

## Inputs

- **Mono signals** at the same sample rate. The model runs at 44.1 kHz; other rates are resampled to it.
- **Playback level:** the reference is played at 65 dB SPL, about conversational speech (`ANNI(level=...)` changes
  this). Both signals get the same gain, so a level change in the test signal counts as a difference.
- **Timing:** a constant delay of up to ±10 ms is removed. If the lengths differ, the longer signal is cut.
- **Length:** clips from 0.5 s to about 10 s, as for PESQ and ViSQOL. Memory grows with length: a 5 s pair takes about
  1 s and 1.5 GB on a laptop CPU. On an NVIDIA GPU, use `ANNI(device="cuda")`.

ANNI warns whenever it changes the inputs (resampling, cutting, removing a delay or a constant offset) and when the
test signal is polarity-inverted. It also warns when the two signals correlate weakly (below 0.9), for example after a
speech codec: the scores may then not be meaningful.

## How it works

1. Both signals are played at the same listening level, and the test signal is aligned to the reference.
2. A filter models the outer and middle ear.
3. An auditory-nerve model turns each signal into the firing rates of 128 nerve fibres tuned from 80 Hz to 16 kHz:
   cochlear (gammatone) filters, inner-hair-cell rectification and smoothing, and a sigmoid firing-rate curve.
4. Two observers compare the firing rates. Each change is weighed against the natural randomness of nerve firing,
   which is what makes loud sounds mask quieter ones. One observer looks at slow changes in firing rate (level and
   colouration), the other at changes in how the rate fluctuates (roughness, flutter).
5. The evidence is added up over fibres and over 200 ms windows. `D_peak` is the largest window, `D_med` the median.

Four constants set the scale so that classic laboratory thresholds score about D = 1: the threshold of hearing and
the 1 dB level difference at 1 kHz (following Taal et al. 2012), noise and tone masking (ITU-R BS.1387), and
amplitude-modulation detection on tones (He et al. 2008) and on noise (O'Connor et al. 2011).

A function-by-function walk-through of the code is in [docs.md](docs.md). For a comparison with PESQ and ViSQOL on
controlled changes, with audio, see [examples/](examples/).

## Evaluation

On [ODAQ](https://zenodo.org/records/10405774), an open dataset of music, film sound and speech over music rated by 26
listeners (210 versions of 30 excerpts; Torcoli et al. 2024), `D_med` orders the versions like the listeners with a
Spearman correlation of 0.85 (95 % interval 0.79–0.89). That is on par with Zimtohrli (0.83) and above ViSQOL (0.65 in
audio mode) and PESQ (0.42). On degraded speech, in the TCD-VoIP dataset (Harte et al. 2015; packet loss, clipping,
echo, competing speakers and noise, rated by 24 listeners), PESQ and ViSQOL follow listeners better.

## Limitations

- D = 1 is calibrated on laboratory sounds (tones and noise). How well it marks the threshold for speech, music and
  watermarks has not been tested with listeners yet.
- It compares the signals moment by moment, so it does not suit audio whose timing or pitch has changed
  (time-stretching, pitch-shifting) or that went through a speech codec.
- It doesn't see energy above 16 kHz or below 80 Hz, very fast fluctuations or fine timing structure.
- Mono only, normal hearing.

## Credits and licence

ANNI is developed by DeepMark Inc. and released under the [MIT License](LICENSE).

The auditory-nerve model is adapted from Mark R. Saddler's
[auditory_machine_learning](https://github.com/msaddler/auditory_machine_learning). If you use ANNI, please also cite:

- Saddler, M. R., & McDermott, J. H. (2024). Models optimized for real-world tasks reveal the task-dependent necessity
  of precise temporal coding in hearing. *Nature Communications*.
  [doi:10.1038/s41467-024-54700-5](https://doi.org/10.1038/s41467-024-54700-5)

Listening-test data:

- Torcoli, M., Wu, C.-W., Dick, S., Williams, P. A., Halimeh, M. M., Wolcott, W., & Habets, E. A. P. (2024). ODAQ: Open
  Dataset of Audio Quality. *Proc. IEEE ICASSP*. [arXiv:2401.00197](https://arxiv.org/abs/2401.00197)
- Harte, N., Gillen, E., & Hines, A. (2015). TCD-VoIP, a research database of degraded speech for assessing quality in
  VoIP applications. *Proc. QoMEX*.

Calibration references:

- Taal, C. H., Hendriks, R. C., & Heusdens, R. (2012). A low-complexity spectro-temporal distortion measure for audio
  processing applications. *IEEE Transactions on Audio, Speech, and Language Processing*, 20(5).
  [doi:10.1109/TASL.2012.2186129](https://doi.org/10.1109/TASL.2012.2186129)
- ITU-R BS.1387: Method for objective measurements of perceived audio quality (PEAQ).
- He, N., Mills, J. H., Ahlstrom, J. B., & Dubno, J. R. (2008). Age-related differences in the temporal modulation
  transfer function with pure-tone carriers. *The Journal of the Acoustical Society of America*, 124(6), 3841–3849.
  [doi:10.1121/1.2998779](https://doi.org/10.1121/1.2998779)
- O'Connor, K. N., Johnson, J. S., Niwa, M., Noriega, N. C., Marshall, E. A., & Sutter, M. L. (2011). Amplitude
  modulation detection as a function of modulation frequency and stimulus duration: comparisons between macaques and
  humans. *Hearing Research*, 277, 37–43. [doi:10.1016/j.heares.2011.03.014](https://doi.org/10.1016/j.heares.2011.03.014)
