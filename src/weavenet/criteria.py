import torch
import torch.nn.functional as F
from .loss import loss_one2one_correlation_exp, loss_one2one_correlation, loss_one2many_penalty, loss_stability, loss_sexequality, loss_egalitarian, loss_balance
from .metric import default_stable_matching_metric

from typing import Dict, Optional, List, Tuple


class _BaseStableMatchingCriteria:
    r"""Internal base for stable-matching criteria classes.

    Subclasses implement :meth:`generate_criterion`; this base supplies the
    surface API (``fairness`` / ``larger_is_better`` / ``base_criterion_names``
    / ``fairness_criterion_name`` / ``metric`` / ``metric_names``) consumed
    by downstream lit modules.

    The defaults for ``base_criterion_names`` and ``metric_names`` reflect
    the universal semantic roles of a stable-matching criterion:
    "encourage 1-to-1 structure" (``loss_one2one``) and "encourage
    stability" (``loss_stability``); plus the corresponding fairness
    metrics. Same-name-different-implementation is the expected
    polymorphism — e.g., :class:`CriteriaStableMatching` fills the
    ``loss_one2one`` slot with one of the ``loss_one2one_*`` functions,
    while :class:`PerAxisSoftmaxCriteria` fills it with the
    matrix-constraint normed-correlation form. Subclasses are free to
    override either list if their loss taxonomy is fundamentally different.

    Args:
       fairness: which fairness metric labels the run for downstream
         best-checkpoint tracking, one of ``'sexequality'``,
         ``'egalitarian'``, ``'balance'``, or ``None``.
       fairness_weight: scalar weight on the fairness term. Passing
         ``0.0`` nulls out ``self.fairness``, so the lit module falls
         back to ``is_success``-based best tracking.
       gate_fairness_loss: if True, the fairness term is gated by the
         satisfaction of the always-on losses. Concrete gating semantics
         are defined by the subclass.
    """

    base_criterion_names: List[str] = ['loss_one2one', 'loss_stability']
    metric_names: List[str] = [
        'is_one2one', 'is_stable', 'is_success', 'num_blocking_pair',
        'sexequality', 'egalitarian', 'balance',
    ]
    # default_stable_matching_metric lives in weavenet.metric and operates on
    # binarized matchings, so it is criterion-independent — exposed here so
    # subclasses inherit the same metric callable without duplication.
    metric = staticmethod(default_stable_matching_metric)

    def __init__(
        self,
        fairness: Optional[str] = 'sexequality',
        fairness_weight: float = 0.1,
        gate_fairness_loss: bool = False,
    ) -> None:
        if fairness_weight == 0.0:
            fairness = None
        self.fairness = fairness
        self.fairness_weight = fairness_weight
        self.gate_fairness_loss = bool(gate_fairness_loss)
        # Lower-is-better for sex-equality cost; higher-is-better otherwise
        # (egalitarian / balance are framed as `-` of the cost, so larger
        # is better in the val/best aggregate sense).
        self.larger_is_better = fairness != 'sexequality'

    def generate_criterion(self):
        raise NotImplementedError(
            "Subclasses of _BaseStableMatchingCriteria must implement generate_criterion()."
        )

    @property
    def fairness_criterion_name(self) -> str:
        return 'loss_{}'.format(self.fairness)


class CriteriaStableMatching(_BaseStableMatchingCriteria):
    def __init__(self,
                 one2one_weight: float = 1.0,
                 stability_weight: float = 0.7,
                 fairness: Optional[str] = 'sexequality',
                 fairness_weight: float = 0.1,
                 loss_one2one: str = 'correlation_exp', # correlation | correlation_exp | maximize_sum
                 gate_fairness_loss: bool = False, # if True, consider fairness loss only when the condition was satisfied.
                ):
        super().__init__(
            fairness=fairness,
            fairness_weight=fairness_weight,
            gate_fairness_loss=gate_fairness_loss,
        )
        self.loss_one2one = loss_one2one
        self.one2one_weight = one2one_weight
        self.stability_weight = stability_weight

    def generate_criterion(self):
        if self.loss_one2one == 'correlation':
            loss_one2one = loss_one2one_correlation
        elif self.loss_one2one == 'correlation_exp':
            loss_one2one = loss_one2one_correlation_exp
        elif self.loss_one2one == 'loss_one2many_penalty':
            loss_one2one = loss_one2many_penalty
        else:
            RuntimeError('Unknown loss_one2one function "{}".'.format(loss_one2one))

        def _criterion_sm(
            m: torch.Tensor,
            sab: torch.Tensor,
            sba_t: torch.Tensor,
            one2one_weight : float = self.one2one_weight,
            stability_weight: float = self.stability_weight,
        ):
            log = {}
            fut_o = torch.jit.fork(loss_one2one,m)#ab, mba_t)
            l_s = loss_stability(m, sab, sba_t)
            l_o = torch.jit.wait(fut_o)

            loss = one2one_weight * l_o
            log['loss_one2one'] = l_o

            loss += stability_weight * l_s
            log['loss_stability'] = l_s

            return loss, log

        if self.fairness is None:
            def criterion_no_fairness(
                m: torch.Tensor,
                sab: torch.Tensor,
                sba_t: torch.Tensor,
                one2one_weight : float = self.one2one_weight,
                stability_weight: float = self.stability_weight,
            ):
                assert(m.size(-1)==1)
                m = m.squeeze(-1).unsqueeze(1)
                loss, log = _criterion_sm(m, sab, sba_t, one2one_weight, stability_weight)
                if m.size(1)==1:
                    return loss, log

                return loss, log
            return criterion_no_fairness


        if self.fairness == 'sexequality':
            loss_fairness = loss_sexequality
        elif self.fairness == 'egalitarian':
            loss_fairness = loss_egalitarian
        elif self.fairness == 'balance':
            loss_fairness = loss_balance
        else:
            RuntimeError('Unknown fairness criterion "{}".'.format(self.fairness))

        def criterion(
            m: torch.Tensor,
            sab: torch.Tensor,
            sba_t: torch.Tensor,
            fairness_weight : float = self.fairness_weight,
            fairness_criterion_name : str = self.fairness_criterion_name,
            gate_fairness_loss : bool = self.gate_fairness_loss,
        ):
            assert(m.size(-1)==1)
            m = m.squeeze(-1).unsqueeze(1) # B, N, M, C=1 -> B, C=1, N, M
            loss, log = _criterion_sm(m, sab, sba_t)

            l = loss_fairness(m, sab, sba_t)

            loss += fairness_weight * l * ((loss.detach()<=0).max(torch.tensor([not gate_fairness_loss], device=l.device)).to(l.dtype))
            # equivalent to ...
            #  if not self.gate_fairness_loss:
            #    loss = fairness_weight * l
            # else:
            #   loss = fairness_weight * l* (loss.detach()<=0)
            log[fairness_criterion_name] = l[:,0] # set it as B, N, M for later fairness - penalty calculation.
            return loss, log


        return criterion


def _normed_correlation_matrix_constraint(
    m: torch.Tensor, p: float = 2.0, epsilon: float = 1e-7
) -> torch.Tensor:
    r"""Doubly-stochastic encouragement on a raw-logit batched matrix.

    .. math::
        d_M = 1 - Z \sum_{ij}
              \frac{e^{m_{ij}}}{\|e^{m_{i,:}}\|_p}
              \cdot
              \frac{e^{m_{ij}}}{\|e^{m_{:,j}}\|_p},
        \quad Z = \frac{N+M}{2NM}

    Returns a per-batch scalar tensor of shape ``(B,)``.
    """
    m_exp = torch.clamp(m, min=epsilon).exp()
    mc_norm = m_exp.norm(p=p, dim=-1, keepdim=True)
    mr_norm = m_exp.norm(p=p, dim=-2, keepdim=True)
    N, M = m.shape[-2:]
    Z = (N + M) / (2 * N * M)
    inner = (m_exp / mc_norm) * (m_exp / mr_norm)
    return 1.0 - inner.sum(dim=(-1, -2)) * Z


def _unstability_score(
    sab: torch.Tensor,
    sba: torch.Tensor,
    m: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    r"""Differentiable proxy for the count of blocking pairs.

    For every column ``c`` (i.e., side-``b`` agent):

    .. math::
        u^a_{i,c} = \sum_j m_{ij}\, \mathrm{ReLU}(s^{ab}_{ic} - s^{ab}_{ij})
        \qquad
        u^b_{c,i} = \sum_k m_{kc}\, \mathrm{ReLU}(s^{ba}_{ci} - s^{ba}_{ck})

    and the per-batch loss is ``sum_{c, i} u^a_{i,c} * u^b_{c,i}``.

    Intuition: ``u^a_{ic}`` is the amount of unhappiness side-``a`` agent
    ``i`` would have if it could trade up to ``c``; ``u^b_{c,i}`` mirrors
    that on side-``b``. Their product is non-zero only where *both* sides
    would prefer to trade — i.e., where a blocking pair would form — so
    minimizing the sum drives the soft assignment ``m`` toward stable
    matchings.

    Shapes:
        - ``sab``: ``(B, N, M)``
        - ``sba``: ``(B, M, N)``
        - ``m``:   ``(B, N, M)`` (soft, typically a per-axis softmax of the
          model output)
        - returns: ``(B,)``
    """
    # sab_diff[b, i, c, j] = sab[b, i, c] - sab[b, i, j]
    sab_diff = (sab.unsqueeze(-1) - sab.unsqueeze(-2)).clamp(min=epsilon)
    # unsab[b, i, c] = sum_j m[b, i, j] * sab_diff[b, i, c, j]
    unsab = (m.unsqueeze(-2) * sab_diff).sum(dim=-1)  # (B, N, M)

    sba_diff = (sba.unsqueeze(-1) - sba.unsqueeze(-2)).clamp(min=epsilon)
    m_T = m.transpose(-1, -2)  # (B, M, N)
    # unsba[b, c, i] = sum_k m_T[b, c, k] * sba_diff[b, c, i, k]
    unsba = (m_T.unsqueeze(-2) * sba_diff).sum(dim=-1)  # (B, M, N)

    unsab_T = unsab.transpose(-1, -2)  # (B, M, N)
    return (unsab_T * unsba).sum(dim=(-1, -2))  # (B,)


def _per_batch_satisfaction(m: torch.Tensor, sab: torch.Tensor, sba: torch.Tensor) -> torch.Tensor:
    """Total satisfaction `sum_a + sum_b` normalized by N. Per-batch shape ``(B,)``."""
    N = sab.shape[-2]
    return ((m * sab).sum(dim=(-1, -2)) + (m.transpose(-1, -2) * sba).sum(dim=(-1, -2))) / N


def _per_batch_balance(m: torch.Tensor, sab: torch.Tensor, sba: torch.Tensor) -> torch.Tensor:
    """Balanced satisfaction `min(sum_a, sum_b) / N`. Per-batch shape ``(B,)``."""
    N = sab.shape[-2]
    sum_a = (m * sab).sum(dim=(-1, -2))
    sum_b = (m.transpose(-1, -2) * sba).sum(dim=(-1, -2))
    return torch.minimum(sum_a, sum_b) / N


def _per_batch_fairness(m: torch.Tensor, sab: torch.Tensor, sba: torch.Tensor) -> torch.Tensor:
    """Sex-equality cost ``|sum_a - sum_b| / N``. Per-batch shape ``(B,)``."""
    N = sab.shape[-2]
    sum_a = (m * sab).sum(dim=(-1, -2))
    sum_b = (m.transpose(-1, -2) * sba).sum(dim=(-1, -2))
    return (sum_a - sum_b).abs() / N


class PerAxisSoftmaxCriteria(_BaseStableMatchingCriteria):
    r"""Criterion that takes raw model logits, softmax-normalizes them along
    each spatial axis independently, computes each loss component on each
    axis-normalized view, and averages the per-axis losses.

    Pairs with :class:`weavenet.layers.MeanAggregator` (raw mean of the two
    stream outputs). The point is to leave probability normalization out of
    the network and inside the loss, so that the model is supervised on
    *both* a row-stochastic interpretation and a column-stochastic
    interpretation of the same raw scores.

    Compare with :class:`CriteriaStableMatching`, which assumes the input
    ``m`` has already been turned into a probability-like matrix by, e.g.,
    :class:`weavenet.layers.DualSoftmaxSqrt`, and computes a single loss
    against that combined view.

    .. math::
        m_c &= \mathrm{softmax}(m,\ \mathrm{dim}=-1) \quad \text{(row-stochastic)} \\
        m_r &= \mathrm{softmax}(m,\ \mathrm{dim}=-2) \quad \text{(column-stochastic)} \\
        L &= \lambda_m\,L_{\text{matrix}}(m, m_c, m_r)
             + \lambda_u\,\tfrac{L_{\text{uns}}(m_c) + L_{\text{uns}}(m_r)}{2}
             + \lambda_s\,\tfrac{-L_{\text{sat}}(m_c) - L_{\text{sat}}(m_r)}{2} \\
            & \quad + \lambda_b\,\tfrac{-L_{\text{bal}}(m_c) - L_{\text{bal}}(m_r)}{2}
             + \lambda_f\,\tfrac{L_{\text{fair}}(m_c) + L_{\text{fair}}(m_r)}{2}

    where :math:`L_{\text{matrix}}` is either the normed-correlation form
    (``constraint_p >= 0``) or ``|m_c - m_r|.mean()`` (``constraint_p < 0``).

    This criterion follows the loss formulation used in the original
    WeaveNet paper (arXiv:2310.12515) verbatim. The first 5 constructor
    args (``one2one_weight``, ``stability_weight``, ``fairness``,
    ``fairness_weight``, ``gate_fairness_loss``) share names and defaults
    with :class:`CriteriaStableMatching` so existing configs can swap
    ``_target_`` between the two without touching other keys; the extra
    paper-specific knobs (``satisfaction_weight``, ``balance_weight``,
    ``constraint_p``) are appended and default to no-op values. Mapping to
    the paper's command-line flags::

        -m  -> one2one_weight        (lambda_m, the matrix constraint)
        -u  -> stability_weight      (lambda_u)
        -s  -> satisfaction_weight   (lambda_s)
        -b  -> balance_weight        (lambda_b)
        -f  -> fairness_weight       (lambda_f)
        -cp -> constraint_p

    Args:
       one2one_weight: weight on the doubly-stochastic / one-to-one
         constraint term (the paper's matrix constraint). Same semantic
         role as in :class:`CriteriaStableMatching`; the underlying
         function differs (this class always uses the normed-correlation
         form, no ``loss_one2one`` dispatch).
       stability_weight: weight on the (differentiable) blocking-pair proxy.
       fairness: which fairness metric labels the run for downstream
         best-checkpoint tracking. Same options as in
         :class:`CriteriaStableMatching`: ``'sexequality'`` (default),
         ``'egalitarian'``, ``'balance'``, or ``None``.
       fairness_weight: weight on the fairness loss; passing ``0.0``
         disables the term and nulls out ``self.fairness``.
       gate_fairness_loss: if True, multiply the fairness term by
         ``(stability_loss <= 0)`` so it only contributes once the
         stability constraint has been satisfied. Same intent as in
         :class:`CriteriaStableMatching` (modulo the per-sample
         instead-of-per-batch granularity there).
       satisfaction_weight: weight on `-(sum_a + sum_b)/N` (maximize total).
       balance_weight: weight on `-min(sum_a, sum_b)/N` (maximize the weaker side).
       constraint_p: p-norm in the matrix-constraint term; pass any
         negative value to fall back to the ``|m_c - m_r|.mean()`` form.
    """

    def __init__(
        self,
        one2one_weight: float = 1.0,
        stability_weight: float = 0.7,
        fairness: Optional[str] = 'sexequality',
        fairness_weight: float = 0.1,
        gate_fairness_loss: bool = False,
        satisfaction_weight: float = 0.0,
        balance_weight: float = 0.0,
        constraint_p: float = 2.0,
    ) -> None:
        # The first 5 args match CriteriaStableMatching's signature so a config
        # can swap `_target_` between the two classes without touching keys.
        # `loss_one2one` is intentionally NOT mirrored: this criterion always
        # uses the matrix-constraint normed-correlation form (paper-spec), so
        # there is nothing to dispatch on.
        super().__init__(
            fairness=fairness,
            fairness_weight=fairness_weight,
            gate_fairness_loss=gate_fairness_loss,
        )
        self.one2one_weight = one2one_weight
        self.stability_weight = stability_weight
        self.satisfaction_weight = satisfaction_weight
        self.balance_weight = balance_weight
        self.constraint_p = constraint_p

    def generate_criterion(self):
        lambda_m = self.one2one_weight
        lambda_u = self.stability_weight
        lambda_s = self.satisfaction_weight
        lambda_b = self.balance_weight
        lambda_f = self.fairness_weight
        constraint_p = self.constraint_p
        fairness = self.fairness
        gate = self.gate_fairness_loss

        def criterion(
            m: torch.Tensor,
            sab: torch.Tensor,
            sba_t: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            assert m.size(-1) == 1, f"expected channel last dim = 1, got {m.shape}"
            m = m.squeeze(-1)            # (B, N, M)
            sba = sba_t.transpose(-1, -2)  # (B, M, N)

            mc = F.softmax(m, dim=-1)    # row-stochastic
            mr = F.softmax(m, dim=-2)    # column-stochastic

            if constraint_p >= 0:
                l_mat = _normed_correlation_matrix_constraint(m, p=constraint_p)
            else:
                l_mat = (mc - mr).abs().mean(dim=(-1, -2))

            l_uns = (_unstability_score(sab, sba, mc) + _unstability_score(sab, sba, mr)) / 2
            l_sat = -(_per_batch_satisfaction(mc, sab, sba) + _per_batch_satisfaction(mr, sab, sba)) / 2
            l_bal = -(_per_batch_balance(mc, sab, sba) + _per_batch_balance(mr, sab, sba)) / 2
            l_fair = (_per_batch_fairness(mc, sab, sba) + _per_batch_fairness(mr, sab, sba)) / 2

            # Mirror CriteriaStableMatching's `gate_fairness_loss` semantics:
            # if gating is on, only apply the fairness term when the
            # stability-side loss has been satisfied (`l_uns <= 0`).
            if gate and (fairness is not None):
                gate_mask = (l_uns.detach() <= 0).to(l_fair.dtype)
                l_fair = l_fair * gate_mask

            total = (
                lambda_m * l_mat
                + lambda_u * l_uns
                + lambda_s * l_sat
                + lambda_b * l_bal
                + lambda_f * l_fair
            )
            log: Dict[str, torch.Tensor] = {
                # Slot the paper's matrix-constraint term and unstability term
                # into the universal `loss_one2one` / `loss_stability` slots
                # declared by `_BaseStableMatchingCriteria.base_criterion_names`.
                "loss_one2one": l_mat,
                "loss_stability": l_uns,
            }
            if fairness == "sexequality":
                log["loss_sexequality"] = l_fair
            elif fairness == "egalitarian":
                log["loss_egalitarian"] = l_sat
            elif fairness == "balance":
                log["loss_balance"] = l_bal
            return total, log

        return criterion
