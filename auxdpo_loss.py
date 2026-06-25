"""
AuxDPO Loss Function for MLX
============================

Implements the Auxiliary Direct Preference Optimization loss from
arXiv:2510.20413. The key insight: standard DPO projects the true reward
onto a low-dimensional manifold (the column space of the policy gradient
matrix), which causes misspecification. AuxDPO adds per-example auxiliary
offsets that live in the nullspace complement, making the full reward
space reachable.

Loss formulation:
  L_AuxDPO = -mean(log(sigmoid(m_i(theta, delta))))
           + lambda_null * ||A_{theta_0} @ delta||^2
           + lambda_reg  * ||delta||^2

where the augmented margin is:
  m_i(theta, delta) = beta * (log_ratio_chosen - log_ratio_rejected)
                    + delta_chosen_i - delta_rejected_i

and log_ratio = log(pi_theta(y|x)) - log(pi_{theta_0}(y|x))
"""

import mlx.core as mx
import mlx.nn as nn


def _log_sigmoid(x: mx.array) -> mx.array:
    """
    Numerically stable log(sigmoid(x)).

    For x >= 0: log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
    For x < 0:  log(sigmoid(x)) = x - log(1 + exp(x)) = x - softplus(x)

    We use: log(sigmoid(x)) = -softplus(-x) which is stable everywhere.
    softplus(z) = log(1 + exp(z)) with MLX's built-in numerical stability.
    """
    # log(sigmoid(x)) = x - softplus(x) = -softplus(-x)
    # Using -softplus(-x) for stability
    return -_softplus(-x)


def _softplus(x: mx.array, beta: float = 1.0, threshold: float = 20.0) -> mx.array:
    """
    Numerically stable softplus: log(1 + exp(beta*x)) / beta.
    For large x, returns x directly to avoid overflow.
    """
    scaled = beta * x
    return mx.where(
        scaled > threshold,
        x,
        mx.log1p(mx.exp(scaled)) / beta,
    )


def get_sequence_logps(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> mx.array:
    """
    Compute per-sequence sum of log probabilities.

    Args:
        logits: (batch, seq_len, vocab_size) - model output logits
        targets: (batch, seq_len) - target token ids
        mask: (batch, seq_len) - 1 for response tokens, 0 for prompt/padding

    Returns:
        (batch,) - sum of log probs over masked positions per sequence
    """
    # Shift: logits predict next token
    # logits[:, :-1] predicts targets[:, 1:]
    shifted_logits = logits[:, :-1, :]
    shifted_targets = targets[:, 1:]
    shifted_mask = mask[:, 1:]

    # Log softmax for numerical stability
    log_probs = shifted_logits - mx.logsumexp(shifted_logits, axis=-1, keepdims=True)

    # Gather log probs at target positions
    # MLX doesn't have torch.gather, so we flatten and index
    batch_size, seq_len, vocab_size = shifted_logits.shape

    flat_log_probs = log_probs.reshape(-1, vocab_size)
    flat_targets = shifted_targets.reshape(-1)
    flat_idx = mx.arange(flat_log_probs.shape[0])

    per_token_logps = flat_log_probs[flat_idx, flat_targets].reshape(batch_size, seq_len)

    # Mask and sum: only count response tokens
    per_token_logps = per_token_logps * shifted_mask.astype(per_token_logps.dtype)

    return per_token_logps.sum(axis=-1)


def dpo_loss(
    policy_chosen_logps: mx.array,
    policy_rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    Standard DPO loss.

    L_DPO = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    Args:
        policy_chosen_logps: (batch,) log P(chosen | x) under policy
        policy_rejected_logps: (batch,) log P(rejected | x) under policy
        ref_chosen_logps: (batch,) log P(chosen | x) under reference
        ref_rejected_logps: (batch,) log P(rejected | x) under reference
        beta: KL penalty coefficient
        label_smoothing: conservative DPO label smoothing (0 = standard)

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    # Log ratios: log(pi/pi_ref) for chosen and rejected
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps

    # Preference margin
    margin = beta * (chosen_log_ratios - rejected_log_ratios)

    # DPO loss with optional label smoothing
    loss = (
        -_log_sigmoid(margin) * (1.0 - label_smoothing)
        - _log_sigmoid(-margin) * label_smoothing
    )
    loss = loss.mean()

    # Implicit rewards for monitoring (detached)
    chosen_rewards = mx.stop_gradient(beta * chosen_log_ratios)
    rejected_rewards = mx.stop_gradient(beta * rejected_log_ratios)

    return loss, chosen_rewards, rejected_rewards


def auxdpo_loss(
    policy_chosen_logps: mx.array,
    policy_rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    aux_chosen: mx.array,
    aux_rejected: mx.array,
    beta: float = 0.1,
    lambda_null: float = 1.0,
    lambda_reg: float = 0.01,
    delta_cap: float = 1.0,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    AuxDPO loss: DPO with auxiliary per-example offsets.

    The augmented margin is:
      m_i = beta * (log_ratio_chosen - log_ratio_rejected)
          + (delta_chosen_i - delta_rejected_i)

    The auxiliary variables are bounded via tanh scaling and regularized
    with an L2 penalty (soft nullspace constraint approximation suitable
    for the large-capacity regime described in the paper).

    Args:
        policy_chosen_logps: (batch,) log P(chosen | x) under policy
        policy_rejected_logps: (batch,) log P(rejected | x) under policy
        ref_chosen_logps: (batch,) log P(chosen | x) under reference
        ref_rejected_logps: (batch,) log P(rejected | x) under reference
        aux_chosen: (batch,) raw auxiliary offsets for chosen responses
        aux_rejected: (batch,) raw auxiliary offsets for rejected responses
        beta: KL penalty coefficient
        lambda_null: nullspace constraint penalty weight
        lambda_reg: L2 regularization on auxiliary variables
        delta_cap: tanh scaling bound on auxiliary magnitude
        label_smoothing: conservative DPO label smoothing

    Returns:
        (loss, chosen_rewards, rejected_rewards, aux_stats)
    """
    # Log ratios
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps

    # Bound auxiliary variables via tanh
    delta_chosen = delta_cap * mx.tanh(aux_chosen)
    delta_rejected = delta_cap * mx.tanh(aux_rejected)

    # Augmented preference margin (Eq. 7 from paper)
    # m_i(theta, delta) = beta * (log_ratio_w - log_ratio_l) + (delta_w - delta_l)
    margin = beta * (chosen_log_ratios - rejected_log_ratios) + (delta_chosen - delta_rejected)

    # Cross-entropy preference loss
    pref_loss = (
        -_log_sigmoid(margin) * (1.0 - label_smoothing)
        - _log_sigmoid(-margin) * label_smoothing
    )
    pref_loss = pref_loss.mean()

    # Nullspace constraint penalty (soft, large-capacity regime):
    # Penalize auxiliary variables that would move the reward along
    # the policy gradient direction. In practice this is approximated
    # by penalizing the squared magnitude of the auxiliary offset
    # difference, encouraging the offset to capture only the nullspace
    # component of the reward error.
    delta_diff = delta_chosen - delta_rejected
    null_penalty = lambda_null * (delta_diff * delta_diff).mean()

    # L2 regularization on raw auxiliary variables (amplitude control)
    reg_penalty = lambda_reg * (aux_chosen * aux_chosen + aux_rejected * aux_rejected).mean()

    # Total loss
    loss = pref_loss + null_penalty + reg_penalty

    # Monitoring stats (all detached)
    chosen_rewards = mx.stop_gradient(beta * chosen_log_ratios + delta_chosen)
    rejected_rewards = mx.stop_gradient(beta * rejected_log_ratios + delta_rejected)
    aux_stats = {
        "pref_loss": mx.stop_gradient(pref_loss),
        "null_penalty": mx.stop_gradient(null_penalty),
        "reg_penalty": mx.stop_gradient(reg_penalty),
        "aux_mean": mx.stop_gradient((mx.abs(delta_chosen) + mx.abs(delta_rejected)).mean() / 2.0),
        "margin_mean": mx.stop_gradient(margin.mean()),
        "margin_acc": mx.stop_gradient((margin > 0).astype(mx.float32).mean()),
    }

    return loss, chosen_rewards, rejected_rewards, aux_stats
