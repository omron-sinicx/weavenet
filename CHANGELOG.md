# Changelog

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
