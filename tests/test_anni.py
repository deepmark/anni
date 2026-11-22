import numpy as np
import pytest
import torch

from anni import ANNI, _core

SR = 44100


@pytest.fixture(scope="module")
def measure():
    return ANNI()


@pytest.fixture(scope="module")
def ref():
    t = np.arange(int(1.5 * SR)) / SR
    rng = np.random.default_rng(0)
    return 0.1 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(len(t))


def noisy(x, amp):
    return x + amp * np.random.default_rng(1).standard_normal(len(x))


def test_identical_signals_score_zero(measure, ref):
    assert measure(ref, ref, sr=SR) == {"D_peak": 0.0, "D_med": 0.0}


def test_regression_values(measure, ref):
    s = measure(ref, noisy(ref, 0.003), sr=SR)
    assert s["D_peak"] == pytest.approx(0.529217541217804, rel=1e-5)
    assert s["D_med"] == pytest.approx(0.48977628350257874, rel=1e-5)


def test_larger_difference_scores_higher(measure, ref):
    d = [measure(ref, noisy(ref, a), sr=SR)["D_peak"] for a in (0.001, 0.003, 0.01)]
    assert d[0] < d[1] < d[2]


def test_fast_amplitude_modulation_is_scored(measure):
    """A 300 Hz amplitude modulation of a 9 kHz tone (20 log m = -20 dB, about the listeners' threshold in Kohlrausch,
    Fassel & Dau 2000) lies above the texture observer's bands; the fast bands pick it up."""
    t = np.arange(int(1.5 * SR)) / SR
    c = 0.05 * np.sin(2 * np.pi * 9000 * t)
    with torch.no_grad():
        px, py = measure._prepare(c, c * (1 + 0.1 * np.cos(2 * np.pi * 300 * t)), SR)
        ob = measure._observer
        bx, fx = measure._nerve.binned_rates(px, ob.bin_n, fast=ob)
        by, fy = measure._nerve.binned_rates(py, ob.bin_n, fast=ob)
        with_fast = float(_core.scores(ob, bx, by, fx=fx, fy=fy)["D_peak"])
        without = float(_core.scores(ob, bx, by)["D_peak"])
    assert 0.4 < with_fast < 1.5
    assert with_fast > 2 * without


def test_level_change_is_scored(measure, ref):
    assert measure(ref, ref * 10 ** (1 / 20), sr=SR)["D_peak"] > 0.5


def test_tensor_input_and_constant_delay(measure, ref):
    y = noisy(ref, 0.003)
    a = measure(ref, y, sr=SR)
    with pytest.warns(UserWarning, match="shifted by"):
        b = measure(torch.tensor(ref), torch.tensor(np.concatenate([np.zeros(20), y[:-20]])), sr=SR)
    assert b["D_med"] == pytest.approx(a["D_med"], rel=0.1)


def test_no_warnings_for_clean_input(measure, ref):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        measure(ref, noisy(ref, 0.003), sr=SR)


def test_other_sample_rate_is_resampled_with_a_warning(measure):
    sr = 48000
    t = np.arange(int(1.5 * sr)) / sr
    x = 0.1 * np.sin(2 * np.pi * 440 * t)
    with pytest.warns(UserWarning, match="resampling"):
        assert measure(x, x * 1.1, sr=sr)["D_peak"] > 0


def test_dc_offset_is_removed_with_a_warning(measure, ref):
    with pytest.warns(UserWarning, match="DC"):
        s = measure(ref, ref + 0.05, sr=SR)
    assert s["D_peak"] < 1e-6


def test_pauses_do_not_change_the_playback_level(ref):
    from anni import _core
    x = torch.tensor(ref)
    padded = torch.cat([torch.zeros(SR), x])
    assert float(_core.pa_gain(padded)) == pytest.approx(float(_core.pa_gain(x)), rel=1e-5)


def test_input_checks(measure, ref):
    with pytest.raises(ValueError):
        measure(np.stack([ref, ref]), np.stack([ref, ref]), sr=SR)
    with pytest.raises(ValueError):
        measure(ref[:1000], ref[:1000], sr=SR)
    with pytest.raises(ValueError, match="silent"):
        measure(np.full(len(ref), 0.3), ref, sr=SR)
    with pytest.warns(UserWarning, match="correlate weakly"):
        measure(ref, np.random.default_rng(2).standard_normal(len(ref)) * 0.05, sr=SR)
    with pytest.warns(UserWarning, match="differ in length"):
        measure(ref, noisy(ref, 0.003)[:-100], sr=SR)
    with pytest.warns(UserWarning, match="polarity-inverted"):
        measure(ref, -noisy(ref, 0.003), sr=SR)


def test_blocks_give_the_same_scores_as_the_whole_signal(measure, ref):
    """Long signals are processed in blocks to save memory; tiny blocks (stressing the block edges) must give the
    scores of the whole-signal computation."""
    x = np.concatenate([ref, ref, ref])
    y = noisy(x, 0.01)
    burst = 0.3 * np.random.default_rng(3).standard_normal(441)
    y[len(ref):len(ref) + 441] += burst - burst.mean()                 # a 10 ms noise burst near a block edge
    ob = measure._observer
    with torch.no_grad():
        px, py = measure._prepare(x, y, SR)
        R = measure._nerve.rates(torch.stack([px, py]))
        whole = _core.scores(ob, ob.bin(R[0]), ob.bin(R[1]), fx=ob.sub_bin_fast(R[0]), fy=ob.sub_bin_fast(R[1]),
                             block_s=1e9)
        bx, fx = measure._nerve.binned_rates(px, ob.bin_n, block_s=0.3, fast=ob)
        by, fy = measure._nerve.binned_rates(py, ob.bin_n, block_s=0.3, fast=ob)
        blocks = _core.scores(ob, bx, by, fx=fx, fy=fy, block_s=0.5)
        fw = ob.sub_bin_fast(R[0])
        assert float((fx - fw).abs().max()) <= 1e-4 * float(fw.abs().max())      # float32 rounding only
    for k in ("D_peak", "D_med"):
        assert float(blocks[k]) == pytest.approx(float(whole[k]), rel=1e-4, abs=1e-4)
