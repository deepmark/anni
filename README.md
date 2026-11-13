# ANNI: Auditory Nerve Noticeability Index

ANNI is a full-reference measure of how noticeable the difference between a reference and a test audio signal is.
It simulates how the auditory nerve responds to each signal and compares the two responses the way an ideal observer
would. The score is in d′ units, the standard unit of detectability: **0 means no difference, and larger values mean
a more noticeable difference.** ANNI is built for small differences near the threshold of hearing, such as audio
watermarks.

> **Status: research prototype (v0.1).** ANNI is calibrated so that D = 1 marks a just-noticeable difference: the
> point at which listeners in classic laboratory experiments start to detect a change. A score below 1 should
> therefore mean the difference is hard to hear, even when comparing directly. This has not yet been confirmed by
> listening tests on real-world audio.

## Install

```bash
pip install deepmark-anni
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
- **Length:** at least 0.5 s; typical test clips are about 10 s long, as for PESQ and ViSQOL. A 5 s pair takes about
  1 s and 0.9 GB on a laptop CPU. Long files are processed in blocks, so memory grows only slowly with length
  (2 min: about 20 s and 1.2 GB). On an NVIDIA GPU, use `ANNI(device="cuda")`.

ANNI warns whenever it changes the inputs (resampling, cutting, removing a delay or a constant offset between them)
and when the test signal is polarity-inverted. It also warns when the two signals correlate weakly (below 0.9), for
example after a speech codec: the scores may then not be meaningful.

## Training loss (experimental)

`ANNILoss` is ANNI as a differentiable loss, for training a model to make changes that are not noticeable (e.g. a watermark embedder). It is zero while the change is at most `target` noticeable and grows quadratically above it.

```python
from anni import ANNILoss

loss_fn = ANNILoss(target=0.5, p=6, level=75.0, device="cuda")
loss = loss_fn(reference, watermarked)    # [batch, time] tensors at 44.1 kHz, lined up sample by sample
loss.backward()
```

| Argument | Meaning |
|---|---|
| `target` | how noticeable the change may be, in d′: 1 is just noticeable (about 3 in 4 correct picks), 0.5 about 64 % |
| `p` | how the 200 ms windows are summarised: 1 is their mean, larger values move towards the most noticeable moment, `float("inf")` is the peak (`D_peak`) |
| `level` | playback level of the reference in dB SPL (default 65). Sounds below the threshold of hearing do not count, so quiet parts of a change can be inaudible at 65 dB and audible when the listener turns the volume up. Train at the loudest level you expect listeners to use, for example 75–80 dB. |
| `polarities` | 2 (default) runs each signal as it is and upside down, as ANNI does; 1 uses about half the memory |

With a finite `p`, the score is a soft maximum that stays below `D_peak` for short changes: with `p=6`, a 10 ms
click reaches about 1.6 times the target at its peak on 1.5 s clips and about twice the target on 10 s clips. Check
`D_peak` with `ANNI` on held-out clips, or use a lower target or a larger `p`.

Unlike `ANNI`, it neither aligns nor resamples the signals, and the signals must be on the loss's device. The
backward pass needs about 175 MB per 1 s example.
It suits models whose changes stay within what ANNI hears: between 80 Hz and 16 kHz (e.g. an embedder working in
100–12000 Hz), in the sound's energy over time and frequency rather than in its phase, and in fluctuations up to
about 180 per second.

## How it works

1. A filter models the outer and middle ear.
2. An auditory-nerve model turns each signal into the firing rates of 128 nerve fibres tuned from 80 Hz to 16 kHz:
   cochlear (gammatone) filters, inner-hair-cell rectification and smoothing, and a sigmoid firing-rate curve.
3. Two observers compare the firing rates. Each change is weighed against the natural randomness of nerve firing,
   which is what makes loud sounds mask quieter ones. One observer looks at slow changes in firing rate (level and
   colouration), the other at changes in how the rate fluctuates (roughness, flutter).
4. The evidence is added up over fibres and over 200 ms windows. `D_peak` is the largest window, `D_med` the median.

Four constants set the scale so that classic laboratory thresholds score about D = 1: the threshold of hearing and
the 1 dB level difference at 1 kHz (following Taal et al. 2012), noise and tone masking (ITU-R BS.1387), and
amplitude-modulation detection on tones (He et al. 2008) and on noise (O'Connor et al. 2011).

A function-by-function walk-through of the code is in [docs.md](https://github.com/deepmark/anni/blob/main/docs.md).
For a comparison with PESQ and ViSQOL on controlled changes, with audio, see
[examples/](https://github.com/deepmark/anni/tree/main/examples).

## Evaluation

On [ODAQ](https://zenodo.org/records/10405774), an open dataset of music, film sound and speech over music rated by 26
listeners (210 versions of 30 excerpts; Torcoli et al. 2024), `D_med` orders the versions like the listeners with a
Spearman correlation of 0.85 (95 % interval 0.79–0.89). That is on par with Zimtohrli (0.83) and above ViSQOL (0.65 in
audio mode) and PESQ (0.42). On degraded speech, in the TCD-VoIP dataset (Harte et al. 2015; packet loss, clipping,
echo, competing speakers and noise, rated by 24 listeners), PESQ and ViSQOL follow listeners better.

## Limitations

- It compares the signals moment by moment, so it does not suit audio whose timing or pitch has changed
  (time-stretching, pitch-shifting) or that went through a speech codec.
- It doesn't see energy above 16 kHz or below 80 Hz, very fast fluctuations or fine timing structure.
- Mono only, normal hearing.

## Credits and licence

ANNI is developed by DeepMark Inc. and released under the
[MIT License](https://github.com/deepmark/anni/blob/main/LICENSE).

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
  [doi:10.1109/TASL.2012.2184753](https://doi.org/10.1109/TASL.2012.2184753)
- ITU-R BS.1387: Method for objective measurements of perceived audio quality (PEAQ).
- He, N., Mills, J. H., Ahlstrom, J. B., & Dubno, J. R. (2008). Age-related differences in the temporal modulation
  transfer function with pure-tone carriers. *The Journal of the Acoustical Society of America*, 124(6), 3841–3849.
  [doi:10.1121/1.2998779](https://doi.org/10.1121/1.2998779)
- O'Connor, K. N., Johnson, J. S., Niwa, M., Noriega, N. C., Marshall, E. A., & Sutter, M. L. (2011). Amplitude
  modulation detection as a function of modulation frequency and stimulus duration: comparisons between macaques and
  humans. *Hearing Research*, 277, 37–43. [doi:10.1016/j.heares.2011.03.014](https://doi.org/10.1016/j.heares.2011.03.014)

## Dedication

ANNI stands for Auditory Nerve Noticeability Index, but the name also carries a personal dedication.
Our wives, Ana Kovačević and Ana Boljević, share the same name. We wanted to dedicate something DeepMark built
to them. Without their support and sacrifice, there would be no DeepMark at all.

By another happy coincidence, our lawyer, sister and huge supporter, Ana Pavlović, shares that name too.
This one is for all three of them. ❤️
