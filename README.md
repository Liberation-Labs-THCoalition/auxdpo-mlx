# AuxDPO for MLX

**First implementation of Auxiliary DPO on Apple Silicon.**

AuxDPO ([arXiv:2510.20413](https://arxiv.org/abs/2510.20413)) fixes a fundamental flaw in standard DPO: the conditional equivalence problem, where DPO can collapse to degenerate solutions that reverse preference ordering or worsen policy reward. AuxDPO introduces per-example auxiliary offsets that absorb the error between parametric and true reward, preventing this collapse.

This implementation runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Why This Exists

Standard DPO is the default preference optimization method for fine-tuning language models. But DPO is a misspecified estimator — when the true reward function can't be exactly realized by the policy class (which is always the case in practice), DPO produces failure modes including preference order reversal and worsening of policy reward.

AuxDPO fixes this. Every model being DPO-trained on Apple Silicon should be using AuxDPO instead. Now it can.

## Quick Start

```bash
pip install mlx mlx-lm

python run_auxdpo.py \
  --model mlx-community/Qwen2.5-0.5B-4bit \
  --data path/to/dpo_pairs.jsonl \
  --output path/to/output/adapter \
  --beta 0.1 \
  --lr 5e-6 \
  --iters 500
```

## Data Format

JSONL with chosen/rejected conversation pairs:

```json
{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## How It Works

### The DPO Problem

Standard DPO loss:
```
L = -log(σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))))
```

When the policy class can't realize the true reward, this estimator is biased. The bias can flip preference ordering — the model learns to prefer the rejected response.

### The AuxDPO Fix

AuxDPO augments the margin with learnable per-example auxiliary offsets:
```
L = -log(σ(β · (log_ratio_chosen - log_ratio_rejected + δ_i)))
```

The auxiliary offset `δ_i` is bounded via `tanh` and regularized to stay near the nullspace of the policy gradient. It absorbs exactly the error that DPO can't handle, preventing degenerate solutions while preserving DPO's computational simplicity.

## Key Implementation Details

- **Reference model without duplication**: Instead of keeping two model copies (memory-prohibitive on Apple Silicon), LoRA B matrices are zeroed for reference passes. The contribution becomes exactly zero (`y = Wx + scale*(x@A)@0 = Wx`), making the policy model act as the reference without extra memory.
- **Metal-accelerated KL divergence**: Uses MLX's native KL divergence kernels for GPU-accelerated loss computation.
- **Bounded auxiliary variables**: `tanh(aux) * delta_cap` prevents auxiliary offsets from dominating the loss.
- **Nullspace regularization**: Soft penalty keeps auxiliary variables near the nullspace of the policy gradient, matching the paper's large-capacity regime.

## Test Suite

```bash
python test_auxdpo.py
```

Five tests covering loss computation, sequence log-probabilities, data loading, LoRA disable/restore mechanism, and full training loop (loss 0.635 → 0.011 in 10 steps).

## Files

| File | Lines | What it does |
|------|-------|-------------|
| `auxdpo_loss.py` | 228 | AuxDPO loss function with auxiliary variable updates |
| `auxdpo_trainer.py` | 514 | Training loop with dual forward passes and LoRA disable/restore |
| `auxdpo_data.py` | 259 | DPO pair data loading with chat template tokenization |
| `run_auxdpo.py` | 263 | CLI entry point with YAML config support |
| `test_auxdpo.py` | 585 | Full test suite |

## Citation

If you use this implementation, please cite the original paper and this repository:

```
@article{auxdpo2025,
  title={Why DPO is a Misspecified Estimator and How to Fix It},
  author={...},
  journal={arXiv preprint arXiv:2510.20413},
  year={2025}
}
```

## License

Apache 2.0

## Support Development

This tool is built and maintained by [Liberation Labs](https://liberationlabs.tech), a worker-owned cooperative building prosocial AI infrastructure. If AuxDPO-MLX saves you time or improves your training:

- ☕ [Support us on Ko-fi](https://ko-fi.com/liberationlabs)
- 🌐 [Learn about our work](https://liberationlabs.tech)
- 💬 [Join our Discord](https://discord.gg/3K2PFnf9se)

Every contribution helps us build AI that belongs to the many, not the few.

---

*Liberation Labs · Worker-owned cooperative · liberationlabs.tech*
*"The daemon watches. The daemon speaks. The daemon does not command."*
