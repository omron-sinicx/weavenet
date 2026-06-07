"""Regression test for the vectorized is_stable / count_blocking_pairs (#477).

Compares the public functions against an independent naive double-loop reference,
including small N where argmax-binarise column collisions/empties are common (the
case a naive per-incumbent shortcut gets wrong — guards against that regression).
"""
import torch

from weavenet.metric import binarize, count_blocking_pairs, is_stable


def _naive(mb, sab, sba_t):
    """Independent reference. Blocking pair (man i, woman c): man i prefers c over
    his match AND, per man a actually assigned to c, woman c prefers i over a."""
    B, N, _ = mb.shape
    total = 0
    stable = torch.ones(B, dtype=torch.bool)
    for b in range(B):
        for i in range(N):
            jm = int(mb[b, i].argmax())
            for c in range(N):
                if sab[b, i, c] > sab[b, i, jm]:
                    for a in range(N):
                        if mb[b, a, c] == 1 and sba_t[b, i, c] > sba_t[b, a, c]:
                            total += 1
                            stable[b] = False
    return total, stable


def test_is_stable_and_count_match_naive():
    g = torch.Generator().manual_seed(0)
    for N, B in [(5, 100), (8, 100), (30, 50)]:  # small N => collisions/empties
        sab = torch.rand(B, N, N, generator=g)
        sba_t = torch.rand(B, N, N, generator=g)
        mb = binarize(torch.rand(B, N, N, generator=g))
        n_ref, s_ref = _naive(mb, sab, sba_t)
        assert torch.equal(is_stable(mb, sab, sba_t).bool(), s_ref), f"is_stable N={N}"
        assert int(count_blocking_pairs(mb, sab, sba_t)) == n_ref, f"count N={N}"


def test_is_stable_consistent_with_count():
    """is_stable must be exactly count_blocking_pairs(per-instance)==0."""
    g = torch.Generator().manual_seed(1)
    N, B = 8, 200
    sab = torch.rand(B, N, N, generator=g)
    sba_t = torch.rand(B, N, N, generator=g)
    mb = binarize(torch.rand(B, N, N, generator=g))
    # batch total 0 => all stable
    if int(count_blocking_pairs(mb, sab, sba_t)) == 0:
        assert bool(is_stable(mb, sab, sba_t).all())
    # per-instance consistency via the naive reference
    _, s_ref = _naive(mb, sab, sba_t)
    assert torch.equal(is_stable(mb, sab, sba_t).bool(), s_ref)
