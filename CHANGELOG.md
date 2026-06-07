# Changelog

## [Unreleased]

## [1.1.1] - 2026-06-05

### Changed
- Dropped the `torch_scatter` dependency. `weavenet.sparse.layers` now uses
  native `Tensor.scatter_reduce` / a native segment-softmax, so no
  version-locked compiled wheel is required (no prebuilt wheel exists for recent
  torch). Behavior is preserved.
- `src/weavenet` is now mypy-clean. `# type: ignore` is used only for the
  inherent dense/sparse API divergence (sparse threads a per-edge `vertex_id`
  tensor where the dense API takes an integer `dim_target`) and for the
  order-based dynamic `forward` dispatch.
- `weavenet.sparse.layers.MaskSelectorBySimilarity` /
  `MaskSelectorRadiusNeighbor` / `MaskSelectorReciprocalNeighbor`: marked as
  **experimental** and now raise `NotImplementedError` on forward. The base
  class's `get_threshold_by_k` references undefined names that suggest this
  code path has never been executed. Use `MaskSelectorByLinearInferenceOr` or
  `MaskSelectorByNorm` instead.
- `weavenet.sparse.layers.MaskSelectorBySimilarity.wrapup`: changed `self.train`
  (method reference, always truthy) to `self.training` (the actual boolean
  flag set by `.train()` / `.eval()`).

### Fixed
- `weavenet.metric.sexequality_cost`: used an undefined `cba` (the parameter is
  `cba_t`) when `pformat != cost`, so the cost conversion was never applied.
- `weavenet.criteria.CriteriaStableMatching`: an unknown `loss_one2one` built a
  `RuntimeError` but never raised it, falling through to `UnboundLocalError`.
  Now raises.
- `weavenet.model.Unit._forward_ean` /
  `SetEncoderPointNetTotalDirectional.forward`: undefined `dim_tar` →
  `dim_target` (would crash on that dispatch path).
- `weavenet.sparse.layers.DualSoftmaxFuzzyLogicAndSp.forward`: wrong signature
  referenced undefined `src_id`/`tar_id`; restored the canonical
  `(xab, src_id, tar_id, xba=None)` so the method can run.
- `weavenet.layers.CrossConcatVertexFeatures`: missing `super().__init__()`,
  in-place mutation of an immutable `torch.Size`, and an inconsistent return;
  now initialises properly and returns a 2-tuple per the `Interactor` contract.
- Removed broken `__main__` scratch blocks and dead nested encoders that
  referenced undefined names.
- `weavenet.sparse.layers._kthlargest_resampling`: K semantics were inverted —
  `drop_rate=0.0` kept only 1 element and `drop_rate=1.0` kept all 100. Now keeps
  `round(N * (1 - drop_rate))` elements as documented. This affected every
  `MaskSelector*` in the live path (`MaskSelectorByLinearInferenceOr`,
  `MaskSelectorByNorm`).
- `weavenet.sparse.layers.SimilarityBasedMaskInference.__init__`: missing
  `super().__init__()` caused `AttributeError` on construction.
- `weavenet.sparse.layers.SparseDenseAdaptor`: index/value selection used
  inconsistent cutoffs (`nonzero` vs `>0.5`); harmless for binary masks but
  caused shape mismatches with non-binary masks. Standardized on `>0.5`.
- `weavenet.sparse.model.MatchingNetSp.forward`: `xba_t` residual update was
  silently dropped due to a `xba` typo (`xba_t_keep, xba = xba_t, xba_t + xba_t_keep`).
- `weavenet.sparse.model.MatchingNetSp._forward_single_stream`: used undefined
  `i` instead of the loop variable `l` in residual gate check.
- `weavenet.sparse.model.UnitSp._forward_ean`: `vertex_idr` typo (missing `id`)
  caused `NameError`.

### Tests
- Added `tests/test_sparse_fixes.py` covering all of the above (15 cases).

## [1.1.0] - 2026-05-26

### Added
- `weavenet.layers.MeanAggregator` — arithmetic mean of two stream
  outputs, raw (no softmax). Paper-recipe-friendly aggregator alternative
  to `DualSoftmaxSqrt`. ([#5](https://github.com/omron-sinicx/weavenet/pull/5))
- `weavenet.criteria.CriteriaPerAxisStableMatching` — sibling of
  `CriteriaStableMatching` that takes raw model logits and applies
  `softmax(., dim=-1)` and `softmax(., dim=-2)` separately inside the
  loss, mirroring the original paper's training recipe.
  ([#6](https://github.com/omron-sinicx/weavenet/pull/6))
- `weavenet.criteria._BaseCriteriaStableMatching` — internal base class
  that factors out the shared interface (`fairness` / `larger_is_better`
  / `base_criterion_names` / `fairness_criterion_name` / `metric` /
  `metric_names`) common to all stable-matching criteria.
  ([#6](https://github.com/omron-sinicx/weavenet/pull/6))
- `weavenet.metric.default_stable_matching_metric` — the metric bundle
  (binarize → is_one2one / is_stable / count_blocking_pairs /
  per-axis fairness costs) is now exposed as a top-level function so it
  can be shared across criteria classes without duplication.
  ([#6](https://github.com/omron-sinicx/weavenet/pull/6))

### Fixed
- `criteria.py`: `CriteriaStableMatching` raised a CPU/GPU device-mismatch
  RuntimeError when `fairness != None` on GPU — the inline
  `torch.tensor([not gate_fairness_loss])` was constructed on CPU and
  mixed into a GPU computation. Now passes `device=l.device`.
  ([#3](https://github.com/omron-sinicx/weavenet/pull/3))
- `model.py`: `MatchingNet.forward` silently dropped the residual
  addition on the side-b stream — a `xba_t` → `xba` typo wrote the
  residualized value to an unused local, while the loop variable kept
  its non-residualized value. Any deep WeaveNet trained with
  `calc_residual` had asymmetric stream depths until this fix.
  ([#4](https://github.com/omron-sinicx/weavenet/pull/4))

## [1.0.1] - 2023-02-06
- Some bug fix.

## [1.0.0] - 2023-02-06
- WeaveNet components for stable matching was released privately.
