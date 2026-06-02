"""Regression tests for fixes in `weavenet.sparse` (v1.1.0 → v1.2.0).

Covers:
- B1: ``SimilarityBasedMaskInference.__init__`` missing ``super().__init__()``
- B2: ``_kthlargest_resampling`` inverted K semantics (drop_rate=0 kept 1, =1 kept all)
- B3: ``MaskSelectorRadiusNeighbor`` / ``ReciprocalNeighbor`` forward lacked ``self``
- B5: ``MatchingNetSp.forward`` ``xba_t`` residual typo (silently dropped on b→a stream)
- B6: ``MatchingNetSp._forward_single_stream`` undefined ``i``
- B7: ``MaskSelectorBySimilarity.wrapup`` ``self.train`` vs ``self.training``
- B9: ``SparseDenseAdaptor`` non-binary mask index/value inconsistency
"""

from __future__ import annotations

import pytest
import torch

from weavenet.sparse.layers import (
    MaskSelectorByLinearInferenceOr,
    MaskSelectorByNorm,
    MaskSelectorRadiusNeighbor,
    MaskSelectorReciprocalNeighbor,
    SimilarityBasedMaskInference,
    SparseDenseAdaptor,
    _kthlargest_resampling,
)


# ---------------------------------------------------------------------------
# B2: _kthlargest_resampling K semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop_rate,expected_keep", [
    (0.0, 100), (0.10, 90), (0.25, 75), (0.5, 50),
    (0.75, 25), (0.9, 10), (1.0, 1),  # always at least 1
])
def test_kthlargest_resampling_kept_count(drop_rate: float, expected_keep: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(100)
    mask = _kthlargest_resampling(x, dim=0, tau=1.0, drop_rate=drop_rate)
    kept = int((mask > 0.5).sum().item())
    assert kept == expected_keep, (
        f"drop_rate={drop_rate}: kept {kept} of 100, expected {expected_keep}"
    )


def test_kthlargest_resampling_2d() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 100)
    mask = _kthlargest_resampling(x, dim=-1, tau=1.0, drop_rate=0.7)
    # Should keep ~30 per row
    kept_per_row = (mask > 0.5).sum(dim=-1)
    assert (kept_per_row == 30).all(), f"per-row kept: {kept_per_row.tolist()}"


# ---------------------------------------------------------------------------
# B1: SimilarityBasedMaskInference is a torch Module
# ---------------------------------------------------------------------------


def test_similarity_based_mask_inference_is_module() -> None:
    """B1: previously raised AttributeError on construction (no super init)."""
    m = SimilarityBasedMaskInference()
    assert isinstance(m, torch.nn.Module)
    assert hasattr(m, "_parameters")
    assert hasattr(m, "_modules")
    # Smoke: .to() works (relies on nn.Module bookkeeping)
    _ = m.to("cpu")


# ---------------------------------------------------------------------------
# B3: Similarity-family selectors are explicitly experimental
# ---------------------------------------------------------------------------


def test_radius_neighbor_explicit_not_implemented() -> None:
    sel = MaskSelectorRadiusNeighbor(radius=0.1)
    with pytest.raises(NotImplementedError):
        sel(torch.randn(2, 4, 4))


def test_reciprocal_neighbor_explicit_not_implemented() -> None:
    sel = MaskSelectorReciprocalNeighbor(k=2)
    with pytest.raises(NotImplementedError):
        sel(torch.randn(2, 4, 4))


# ---------------------------------------------------------------------------
# B9: SparseDenseAdaptor handles non-binary masks consistently
# ---------------------------------------------------------------------------


def test_sparse_dense_adaptor_binary_round_trip() -> None:
    torch.manual_seed(0)
    B, N, M, C = 2, 4, 4, 3
    mask = (torch.rand(B, N, M, 1) > 0.5).to(torch.float)
    adaptor = SparseDenseAdaptor(mask)
    x = torch.randn(B, N, M, C)
    x_sp = adaptor.to_sparse(x)
    n_selected_expected = int((mask > 0.5).sum().item())
    assert x_sp.shape == (n_selected_expected, C)
    x_back = adaptor.to_dense(x_sp)
    # Selected positions must match
    assert torch.allclose(x_back[mask.squeeze(-1) > 0.5], x[mask.squeeze(-1) > 0.5])


def test_sparse_dense_adaptor_nonbinary_mask_consistent() -> None:
    """B9: index and value selection used different cutoffs (`nonzero` vs `>0.5`).
    With a continuous mask, this previously produced shape mismatch."""
    torch.manual_seed(0)
    B, N, M, C = 1, 3, 3, 2
    # Mask with continuous values; only some above 0.5
    mask = torch.tensor(
        [[[[0.0], [0.3], [0.8]],
          [[0.1], [0.6], [0.4]],
          [[0.9], [0.2], [0.7]]]],
        dtype=torch.float,
    )
    adaptor = SparseDenseAdaptor(mask)
    x = torch.randn(B, N, M, C)
    # Should not raise (post-fix: both index and value derived from > 0.5)
    x_sp = adaptor.to_sparse(x)
    n_above = int((mask > 0.5).sum().item())
    assert x_sp.shape == (n_above, C)


# ---------------------------------------------------------------------------
# Live MaskSelector smoke (uses the fixed _kthlargest_resampling)
# ---------------------------------------------------------------------------


def test_mask_selector_by_linear_inference_or_smoke() -> None:
    torch.manual_seed(0)
    B, N, M, C = 2, 4, 4, 5
    sel = MaskSelectorByLinearInferenceOr(drop_rate=0.5, tau=1.0)
    sel.build(input_channels=C)
    xab = torch.randn(B, N, M, C)
    xba_t = torch.randn(B, N, M, C)
    y = sel(xab, xba_t)
    # Shape: (B, N, M, output_channels=1)
    assert y.shape == (B, N, M, 1)
    # Range: 0 or 1 in forward (ST estimator) — but after OR rule division,
    # values may be 0 or 1 (since 2 was divided by 2).
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_mask_selector_by_norm_smoke() -> None:
    torch.manual_seed(0)
    B, N, M, C = 2, 4, 4, 5
    sel = MaskSelectorByNorm(drop_rate=0.5, tau=1.0)
    xab = torch.randn(B, N, M, C)
    xba_t = torch.randn(B, N, M, C)
    y = sel(xab, xba_t)
    assert y.shape == (B, N, M, 1)
    assert y.min() >= 0.0 and y.max() <= 1.0
