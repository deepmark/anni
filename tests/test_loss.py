import numpy as np
import pytest
import torch

from anni import ANNI, ANNILoss

SR = 44100


@pytest.fixture(scope="module")
def ref():
    t = np.arange(int(1.5 * SR)) / SR
    rng = np.random.default_rng(0)
    x = 0.1 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(len(t))
    return torch.tensor(x, dtype=torch.float32)


def noisy(x, amp, seed=1):
    return x + amp * torch.tensor(np.random.default_rng(seed).standard_normal(len(x)), dtype=torch.float32)


def test_identical_signals_give_zero_loss_and_zero_gradient(ref):
    y = ref.clone().requires_grad_(True)
    loss = ANNILoss(target=0.0)(ref, y)
    loss.backward()
    assert float(loss) < 1e-10
    assert torch.isfinite(y.grad).all() and float(y.grad.abs().max()) < 1e-6


def test_silent_stretches_give_finite_gradients(ref):
    x = ref.clone()
    x[SR // 2:SR] = 0.0                                                 # a digital pause
    y = noisy(x, 0.003).requires_grad_(True)
    ANNILoss(target=0.0)(x, y).backward()
    assert torch.isfinite(y.grad).all() and float(y.grad.abs().max()) > 0


def test_the_peak_matches_the_measure(ref):
    """With p=inf and both polarities, the loss's score is ANNI's D_peak (the signals are already aligned)."""
    y = noisy(ref, 0.003)
    d_peak = ANNI()(ref.numpy(), y.numpy(), sr=SR)["D_peak"]
    with torch.no_grad():
        score = float(ANNILoss(p=float("inf")).soft_score(ref, y)[0])
    assert score == pytest.approx(d_peak, rel=1e-4)


def test_p_moves_the_score_from_the_mean_to_the_peak(ref):
    y = noisy(ref, 0.003)
    with torch.no_grad():
        w = ANNILoss().window_scores(ref, y)[0]
        s = [float(ANNILoss(p=p).soft_score(ref, y)[0]) for p in (1, 2, 6, float("inf"))]
    assert s[0] == pytest.approx(float(w.mean()), rel=1e-4)
    assert s[-1] == pytest.approx(float(w.max()), rel=1e-4)
    assert s[0] < s[1] < s[2] < s[3]


def test_target_sets_where_the_loss_starts(ref):
    y = noisy(ref, 0.003)
    with torch.no_grad():
        score = float(ANNILoss().soft_score(ref, y)[0])
        assert float(ANNILoss(target=score + 0.1)(ref, y)) == 0.0
        assert float(ANNILoss(target=score - 0.1)(ref, y)) == pytest.approx(0.01, rel=1e-3)


def test_a_batch_gives_the_scores_of_its_examples(ref):
    x = torch.stack([ref, 0.5 * ref])
    y = torch.stack([noisy(ref, 0.003), noisy(0.5 * ref, 0.01, seed=2)])
    loss_fn = ANNILoss()
    with torch.no_grad():
        both = loss_fn.soft_score(x, y)
        one = torch.cat([loss_fn.soft_score(x[i], y[i]) for i in range(2)])
    assert torch.allclose(both, one, rtol=1e-4)


def test_gradient_descent_lowers_anni(ref):
    """A few steps on the loss make the change less noticeable by ANNI's own measure."""
    delta = (0.01 * torch.tensor(np.random.default_rng(3).standard_normal(len(ref)), dtype=torch.float32))
    delta.requires_grad_(True)
    loss_fn, measure = ANNILoss(target=0.0, p=6), ANNI()
    before = measure(ref.numpy(), (ref + delta).detach().numpy(), sr=SR)["D_peak"]
    opt = torch.optim.Adam([delta], lr=1e-3)
    for _ in range(25):
        opt.zero_grad()
        loss_fn(ref, ref + delta).backward()
        opt.step()
    after = measure(ref.numpy(), (ref + delta).detach().numpy(), sr=SR)["D_peak"]
    assert after < 0.7 * before


def test_input_checks(ref):
    loss_fn = ANNILoss()
    with pytest.raises(ValueError, match="44100"):
        loss_fn(ref, ref, sr=48000)
    with pytest.raises(ValueError, match="same shape"):
        loss_fn(ref, ref[:-1])
    with pytest.raises(ValueError, match="0.5 s"):
        loss_fn(ref[:1000], ref[:1000])
    with pytest.raises(ValueError, match="polarities"):
        ANNILoss(polarities=3)
