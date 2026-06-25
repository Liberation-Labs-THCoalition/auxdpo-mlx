"""
AuxDPO Trainer for MLX
======================

Training loop for DPO/AuxDPO on Apple Silicon via MLX.

Key difference from SFT: each training step requires FOUR forward passes:
  1. Policy model on chosen sequences
  2. Policy model on rejected sequences
  3. Reference model on chosen sequences (no gradients)
  4. Reference model on rejected sequences (no gradients)

Reference model trick: instead of keeping two copies of the model, we
disable LoRA adapter weights for the reference pass (revealing the frozen
base model) and re-enable them for the policy pass.
"""

import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .auxdpo_loss import auxdpo_loss, dpo_loss, get_sequence_logps
from .auxdpo_data import DPODataset, iterate_dpo_batches


@dataclass
class AuxDPOArgs:
    """Training arguments for AuxDPO."""
    # Training
    batch_size: int = 1
    iters: int = 500
    learning_rate: float = 5e-6
    aux_lr: float = 5e-3
    beta: float = 0.1
    label_smoothing: float = 0.0

    # AuxDPO specific
    use_auxdpo: bool = True
    lambda_null: float = 1.0
    lambda_reg: float = 0.01
    delta_cap: float = 1.0

    # Model
    max_seq_length: int = 2048
    grad_checkpoint: bool = False

    # Logging and saving
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    adapter_file: str = "adapters.safetensors"

    # Validation
    val_batches: int = 25


def _clear_cache(threshold: int = 0):
    """Clear MLX cache if it grows too large."""
    if threshold > 0 and mx.get_cache_memory() > threshold:
        mx.clear_cache()


def _get_model_logps(model, input_ids, mask):
    """
    Run forward pass and compute per-sequence log probabilities.

    Args:
        model: the language model
        input_ids: (batch, seq_len) token ids
        mask: (batch, seq_len) response mask (1 for response tokens)

    Returns:
        (batch,) sum of log probs over response tokens
    """
    logits = model(input_ids)
    return get_sequence_logps(logits, input_ids, mask)


def _disable_lora(model):
    """
    Temporarily zero out LoRA contributions for reference model pass.

    Returns a dict of saved LoRA state to restore later. Rather than
    removing layers (expensive), we zero the lora_b matrices which
    makes the LoRA contribution exactly 0: y = Wx + scale*(x@A)@B
    and B=0 => LoRA output is 0.
    """
    saved = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_b") and hasattr(module, "lora_a"):
            saved[name] = {
                "lora_a": module.lora_a,
                "lora_b": module.lora_b,
            }
            # Zero B matrix — makes LoRA contribution exactly 0
            module.lora_b = mx.zeros_like(module.lora_b)
    return saved


def _restore_lora(model, saved_state):
    """Restore LoRA weights after reference pass."""
    for name, state in saved_state.items():
        # Navigate to the module by name and set attributes directly
        parts = name.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        module.lora_a = state["lora_a"]
        module.lora_b = state["lora_b"]


class AuxDPOTrainer:
    """
    Trainer for AuxDPO / DPO on MLX with LoRA.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset: DPODataset,
        val_dataset: Optional[DPODataset] = None,
        args: AuxDPOArgs = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.args = args or AuxDPOArgs()

        # Policy optimizer (LoRA weights only)
        self.optimizer = opt.Adam(learning_rate=self.args.learning_rate)

        # Auxiliary variables: one pair (chosen, rejected) per training example
        n_train = len(train_dataset)
        if self.args.use_auxdpo:
            self.aux_chosen = mx.zeros((n_train,))
            self.aux_rejected = mx.zeros((n_train,))
            self.aux_optimizer = opt.Adam(learning_rate=self.args.aux_lr)
            print(f"AuxDPO mode: {n_train} auxiliary variable pairs initialized")
        else:
            self.aux_chosen = None
            self.aux_rejected = None
            print("Standard DPO mode (no auxiliary variables)")

        if mx.metal.is_available():
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    def _compute_loss(self, batch, example_indices=None):
        """
        Compute DPO or AuxDPO loss for a batch.

        This does all four forward passes:
          1-2. Policy model on chosen + rejected
          3-4. Reference model on chosen + rejected
        """
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        chosen_mask = batch["chosen_mask"]
        rejected_mask = batch["rejected_mask"]

        # --- Policy forward passes (with LoRA) ---
        self.model.train()
        policy_chosen_logps = _get_model_logps(self.model, chosen_ids, chosen_mask)
        policy_rejected_logps = _get_model_logps(self.model, rejected_ids, rejected_mask)

        # --- Reference forward passes (without LoRA) ---
        saved_lora = _disable_lora(self.model)
        self.model.eval()

        ref_chosen_logps = mx.stop_gradient(
            _get_model_logps(self.model, chosen_ids, chosen_mask)
        )
        ref_rejected_logps = mx.stop_gradient(
            _get_model_logps(self.model, rejected_ids, rejected_mask)
        )

        # Restore LoRA weights
        _restore_lora(self.model, saved_lora)
        self.model.train()

        # --- Compute loss ---
        if self.args.use_auxdpo and example_indices is not None:
            # Gather auxiliary variables for this batch
            aux_c = self.aux_chosen[example_indices]
            aux_r = self.aux_rejected[example_indices]

            loss, chosen_rewards, rejected_rewards, aux_stats = auxdpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                aux_chosen=aux_c,
                aux_rejected=aux_r,
                beta=self.args.beta,
                lambda_null=self.args.lambda_null,
                lambda_reg=self.args.lambda_reg,
                delta_cap=self.args.delta_cap,
                label_smoothing=self.args.label_smoothing,
            )
            return loss, chosen_rewards, rejected_rewards, aux_stats
        else:
            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                ref_chosen_logps=ref_chosen_logps,
                ref_rejected_logps=ref_rejected_logps,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
            )
            return loss, chosen_rewards, rejected_rewards, {}

    def train(self):
        """Main training loop."""
        args = self.args
        model = self.model

        print(f"Starting {'AuxDPO' if args.use_auxdpo else 'DPO'} training...")
        print(f"  Iterations: {args.iters}")
        print(f"  Beta: {args.beta}")
        print(f"  LR: {args.learning_rate}")
        if args.use_auxdpo:
            print(f"  Aux LR: {args.aux_lr}")
            print(f"  Lambda null: {args.lambda_null}")
            print(f"  Lambda reg: {args.lambda_reg}")
            print(f"  Delta cap: {args.delta_cap}")

        # Training state
        loss_value_and_grad = nn.value_and_grad(model, self._loss_fn_for_grad)
        running_loss = 0.0
        running_acc = 0.0
        train_time = 0.0

        # Batch iterator with example index tracking
        batch_iter = self._make_batch_iterator(loop=True)

        for it in range(1, args.iters + 1):
            tic = time.perf_counter()

            # Validation
            if self.val_dataset and (
                it == 1 or it % args.steps_per_eval == 0 or it == args.iters
            ):
                val_loss, val_acc = self.evaluate()
                val_time = time.perf_counter() - tic
                print(
                    f"Iter {it}: Val loss {val_loss:.4f}, "
                    f"Val acc {val_acc:.3f}, "
                    f"Val took {val_time:.3f}s"
                )
                tic = time.perf_counter()

            # Get batch
            batch, example_indices = next(batch_iter)

            # Forward + backward for policy weights
            (loss_val, aux_stats), grad = loss_value_and_grad(
                model, batch, example_indices
            )

            # Update policy (LoRA) weights
            self.optimizer.update(model, grad)

            # Update auxiliary variables (separate optimization step)
            if args.use_auxdpo and example_indices is not None:
                self._update_aux_variables(batch, example_indices)

            # Evaluate lazy computation
            mx.eval(model.parameters(), self.optimizer.state)
            if args.use_auxdpo:
                mx.eval(self.aux_chosen, self.aux_rejected)

            _clear_cache()

            step_time = time.perf_counter() - tic
            train_time += step_time

            # Accumulate stats
            margin_acc = aux_stats.get("margin_acc", mx.array(0.0))
            if isinstance(margin_acc, mx.array):
                margin_acc = margin_acc.item()
            running_loss += loss_val.item() if isinstance(loss_val, mx.array) else loss_val
            running_acc += margin_acc

            # Report
            if it % args.steps_per_report == 0 or it == args.iters:
                avg_loss = running_loss / args.steps_per_report
                avg_acc = running_acc / args.steps_per_report
                it_sec = args.steps_per_report / train_time
                peak_mem = mx.get_peak_memory() / 1e9

                report = (
                    f"Iter {it}: Loss {avg_loss:.4f}, "
                    f"Acc {avg_acc:.3f}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB"
                )
                if args.use_auxdpo and aux_stats:
                    aux_mean = aux_stats.get("aux_mean", mx.array(0.0))
                    if isinstance(aux_mean, mx.array):
                        aux_mean = aux_mean.item()
                    report += f", Aux |delta| {aux_mean:.4f}"

                print(report)
                running_loss = 0.0
                running_acc = 0.0
                train_time = 0.0

            # Save checkpoint
            if it % args.steps_per_save == 0:
                self._save_checkpoint(it)

        # Final save
        self._save_checkpoint(args.iters, final=True)

    def _loss_fn_for_grad(self, model, batch, example_indices):
        """
        Wrapper for nn.value_and_grad — returns (loss, aux_stats).
        The grad is only computed w.r.t. model parameters (first arg).
        """
        loss, _, _, aux_stats = self._compute_loss(batch, example_indices)
        return loss, aux_stats

    def _update_aux_variables(self, batch, example_indices):
        """
        Update auxiliary variables via gradient descent.

        The aux variables are not model parameters, so we compute their
        gradients separately and apply an Adam update.
        """
        def aux_loss_fn(aux_c, aux_r):
            chosen_ids = batch["chosen_ids"]
            rejected_ids = batch["rejected_ids"]
            chosen_mask = batch["chosen_mask"]
            rejected_mask = batch["rejected_mask"]

            # Policy logps (no grad needed here, just for aux update)
            policy_chosen_logps = mx.stop_gradient(
                _get_model_logps(self.model, chosen_ids, chosen_mask)
            )
            policy_rejected_logps = mx.stop_gradient(
                _get_model_logps(self.model, rejected_ids, rejected_mask)
            )

            # Ref logps
            saved_lora = _disable_lora(self.model)
            ref_chosen_logps = mx.stop_gradient(
                _get_model_logps(self.model, chosen_ids, chosen_mask)
            )
            ref_rejected_logps = mx.stop_gradient(
                _get_model_logps(self.model, rejected_ids, rejected_mask)
            )
            _restore_lora(self.model, saved_lora)

            loss, _, _, _ = auxdpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                aux_c, aux_r,
                beta=self.args.beta,
                lambda_null=self.args.lambda_null,
                lambda_reg=self.args.lambda_reg,
                delta_cap=self.args.delta_cap,
            )
            return loss

        # Compute gradients w.r.t. auxiliary variables
        aux_c = self.aux_chosen[example_indices]
        aux_r = self.aux_rejected[example_indices]

        grad_fn = mx.grad(aux_loss_fn, argnums=(0, 1))
        grad_c, grad_r = grad_fn(aux_c, aux_r)

        # Simple gradient descent update (Adam would require per-variable state
        # management that's complex with dynamic indexing; SGD is sufficient
        # since aux variables converge quickly)
        aux_lr = self.args.aux_lr
        new_c = aux_c - aux_lr * grad_c
        new_r = aux_r - aux_lr * grad_r

        # Scatter updated values back
        self.aux_chosen = self.aux_chosen.at[example_indices].add(new_c - aux_c)
        self.aux_rejected = self.aux_rejected.at[example_indices].add(new_r - aux_r)

    def _make_batch_iterator(self, loop=True):
        """
        Yield (batch_dict, example_indices) tuples.

        example_indices tracks which training examples are in each batch,
        needed for indexing into the per-example auxiliary variables.
        """
        dataset = self.train_dataset
        n = len(dataset)
        batch_size = self.args.batch_size
        max_seq = self.args.max_seq_length

        if n < batch_size:
            raise ValueError(f"Dataset ({n}) smaller than batch_size ({batch_size})")

        # Sort by length for efficiency
        idx = sorted(range(n), key=lambda i: dataset.itemlen(i))
        batch_indices = [
            idx[i:i + batch_size]
            for i in range(0, n - batch_size + 1, batch_size)
        ]

        rng = np.random.RandomState(42)

        while True:
            order = rng.permutation(len(batch_indices))
            for bi in order:
                indices = batch_indices[bi]
                examples = [dataset[j] for j in indices]

                # Collate
                from .auxdpo_data import collate_dpo_batch
                pad_id = dataset.tokenizer.pad_token_id or 0
                batch = collate_dpo_batch(examples, pad_token_id=pad_id)

                # Truncate to max_seq_length
                for key in ["chosen_ids", "rejected_ids", "chosen_mask", "rejected_mask"]:
                    if batch[key].shape[1] > max_seq:
                        batch[key] = batch[key][:, :max_seq]

                yield batch, mx.array(indices)

            if not loop:
                break

    def evaluate(self):
        """Run evaluation on validation set."""
        if self.val_dataset is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in iterate_dpo_batches(
            self.val_dataset,
            batch_size=self.args.batch_size,
            max_seq_length=self.args.max_seq_length,
            loop=False,
        ):
            loss, chosen_r, rejected_r, aux_stats = self._compute_loss(batch)
            acc = (chosen_r > rejected_r).astype(mx.float32).mean()

            mx.eval(loss, acc)
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

            if self.args.val_batches > 0 and n_batches >= self.args.val_batches:
                break

        self.model.train()

        if n_batches == 0:
            return 0.0, 0.0

        return total_loss / n_batches, total_acc / n_batches

    def _save_checkpoint(self, iteration: int, final: bool = False):
        """Save adapter weights and auxiliary variables."""
        adapter_path = Path(self.args.adapter_file)
        adapter_dir = adapter_path.parent
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(str(adapter_path), adapter_weights)

        # Save numbered checkpoint
        if not final:
            checkpoint = adapter_dir / f"{iteration:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)

        # Save auxiliary variables if using AuxDPO
        if self.args.use_auxdpo and self.aux_chosen is not None:
            aux_path = adapter_dir / "aux_variables.npz"
            np.savez(
                str(aux_path),
                aux_chosen=np.array(self.aux_chosen.tolist()),
                aux_rejected=np.array(self.aux_rejected.tolist()),
            )

        # Save training config
        config = {
            "method": "auxdpo" if self.args.use_auxdpo else "dpo",
            "beta": self.args.beta,
            "learning_rate": self.args.learning_rate,
            "aux_lr": self.args.aux_lr if self.args.use_auxdpo else None,
            "lambda_null": self.args.lambda_null if self.args.use_auxdpo else None,
            "lambda_reg": self.args.lambda_reg if self.args.use_auxdpo else None,
            "delta_cap": self.args.delta_cap if self.args.use_auxdpo else None,
            "iteration": iteration,
        }
        config_path = adapter_dir / "auxdpo_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        label = "final" if final else f"iter {iteration}"
        print(f"Saved checkpoint ({label}) to {adapter_dir}")
