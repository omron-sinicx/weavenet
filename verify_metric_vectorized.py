#!/usr/bin/env python3
"""Regression test for the vectorized weavenet.metric.is_stable /
count_blocking_pairs (#477). Compares the public (vectorized) functions against
an independent, obviously-correct naive double-loop reference on real testset
instances (binarised). Guards future edits even though the original fork-based
implementation was replaced. Run: `python verify_metric_vectorized.py`.
"""
import sys
import time

import torch

sys.path.insert(0, "/workspace/external/weavenet/src")
from weavenet.metric import binarize, count_blocking_pairs, is_stable  # noqa: E402

TESTSET = "/workspace/data/shared/experiments/2026-06-05_388-extraction-align/testset_n30_1000.pt"


def naive_count_and_stable(mb, sab, sba_t):
    """Independent reference: a blocking pair (man i, woman c) needs man i to
    prefer c over his match AND, per man a actually assigned to c, woman c to
    prefer i over a. Returns (total_count_over_batch, per_instance_stable bool)."""
    B, N, _ = mb.shape
    total = 0
    stable = torch.ones(B, dtype=torch.bool)
    for b in range(B):
        for i in range(N):
            jm = int(mb[b, i].argmax())  # man i's match (one-hot row)
            for c in range(N):
                if sab[b, i, c] > sab[b, i, jm]:
                    for a in range(N):
                        if mb[b, a, c] == 1 and sba_t[b, i, c] > sba_t[b, a, c]:
                            total += 1
                            stable[b] = False
    return total, stable


def main() -> int:
    data = torch.load(TESTSET, map_location="cpu", weights_only=False)
    rng = torch.Generator().manual_seed(0)
    stable_ok = count_ok = True
    for sab, sba_t in data[:5]:  # naive is O(N^3); a few batches suffice
        sab2 = sab.squeeze(-1).contiguous()
        sba2 = sba_t.squeeze(-1).contiguous()
        mb = binarize(torch.rand(sab2.shape, generator=rng))
        n_ref, s_ref = naive_count_and_stable(mb, sab2, sba2)
        stable_ok &= torch.equal(is_stable(mb, sab2, sba2).bool(), s_ref)
        count_ok &= int(count_blocking_pairs(mb, sab2, sba2)) == n_ref
    print(f"is_stable vs naive:            {stable_ok}")
    print(f"count_blocking_pairs vs naive: {count_ok}")
    assert stable_ok and count_ok, "vectorized metric diverged from naive reference"

    # speed (full 1000 instances)
    t0 = time.perf_counter()
    for sab, sba_t in data:
        sab2, sba2 = sab.squeeze(-1).contiguous(), sba_t.squeeze(-1).contiguous()
        mb = binarize(torch.rand(sab2.shape, generator=torch.Generator().manual_seed(1)))
        is_stable(mb, sab2, sba2)
        count_blocking_pairs(mb, sab2, sba2)
    print(f"vectorized is_stable+count, 1000 inst: {time.perf_counter() - t0:.3f}s")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
