#!/usr/bin/env python3
"""
AuxDPO Training CLI for MLX
============================

First known implementation of AuxDPO on Apple Silicon.
Reference: arXiv:2510.20413

Usage:
    python run_auxdpo.py \
      --model /path/to/model \
      --data /path/to/dpo_pairs.jsonl \
      --output /path/to/output/adapter \
      --beta 0.1 \
      --lr 5e-6 \
      --iters 500

    python run_auxdpo.py --config config/ayni_consent_auxdpo.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="AuxDPO: Auxiliary Direct Preference Optimization on MLX"
    )

    # Config file (overrides individual args)
    parser.add_argument("--config", type=str, help="YAML config file path")

    # Model
    parser.add_argument("--model", type=str, help="Path to MLX model directory")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to existing LoRA adapter to resume from")
    parser.add_argument("--data", type=str, help="Path to DPO pairs JSONL file")
    parser.add_argument("--output", type=str, default="./auxdpo-adapter",
                        help="Output directory for trained adapter")

    # Training hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL penalty coefficient)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate for policy (LoRA) weights")
    parser.add_argument("--aux-lr", type=float, default=5e-3,
                        help="Learning rate for auxiliary variables")
    parser.add_argument("--iters", type=int, default=500,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Maximum sequence length")

    # AuxDPO specific
    parser.add_argument("--no-aux", action="store_true",
                        help="Disable auxiliary variables (standard DPO)")
    parser.add_argument("--lambda-null", type=float, default=1.0,
                        help="Nullspace constraint penalty weight")
    parser.add_argument("--lambda-reg", type=float, default=0.01,
                        help="L2 regularization on auxiliary variables")
    parser.add_argument("--delta-cap", type=float, default=1.0,
                        help="Tanh bound on auxiliary variable magnitude")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for DPO loss")

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-scale", type=float, default=2.0,
                        help="LoRA scale (alpha / rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--num-layers", type=int, default=16,
                        help="Number of layers to apply LoRA")

    # Logging and saving
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--report-every", type=int, default=10,
                        help="Report loss every N steps")
    parser.add_argument("--eval-every", type=int, default=200,
                        help="Evaluate every N steps")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split fraction")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Use gradient checkpointing")

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # Try relative to script directory
            config_path = Path(__file__).parent / args.config
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from {config_path}")
        else:
            print(f"[ERROR] Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)

    # Merge: config file < CLI args (CLI overrides config)
    def get(key, cli_val, default=None):
        if cli_val is not None and cli_val != default:
            return cli_val
        return config.get(key, cli_val if cli_val is not None else default)

    model_path = get("model", args.model)
    adapter_path = get("adapter_path", args.adapter)
    data_path = get("data", args.data)
    output_path = get("output", args.output, "./auxdpo-adapter")

    if not model_path:
        print("[ERROR] --model is required", file=sys.stderr)
        sys.exit(1)
    if not data_path:
        print("[ERROR] --data is required", file=sys.stderr)
        sys.exit(1)

    # Set seed
    seed = get("seed", args.seed, 42)
    mx.random.seed(seed)

    # LoRA parameters
    lora_config = config.get("lora_parameters", {})
    lora_rank = lora_config.get("rank", args.lora_rank)
    lora_scale = lora_config.get("scale", args.lora_scale)
    lora_dropout = lora_config.get("dropout", args.lora_dropout)
    num_layers = get("num_layers", args.num_layers, 16)

    print("=" * 60)
    print("AuxDPO Training on MLX")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print()

    # ---- Load model and tokenizer ----
    from mlx_lm import load as mlx_load
    print("Loading model and tokenizer...")
    model, tokenizer = mlx_load(model_path)
    print(f"Model loaded: {type(model).__name__}")

    # ---- Apply LoRA ----
    from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

    lora_params = {
        "rank": lora_rank,
        "scale": lora_scale,
        "dropout": lora_dropout,
    }

    linear_to_lora_layers(model, num_layers, lora_params)

    # Load existing adapter weights if resuming
    if adapter_path:
        adapter_file = Path(adapter_path)
        if adapter_file.is_dir():
            adapter_file = adapter_file / "adapters.safetensors"
        if adapter_file.exists():
            model.load_weights(str(adapter_file), strict=False)
            print(f"Loaded existing adapter from {adapter_file}")
        else:
            print(f"[WARNING] Adapter file not found: {adapter_file}, starting fresh")

    print_trainable_parameters(model)

    # Freeze non-LoRA parameters
    model.freeze()
    for name, module in model.named_modules():
        if hasattr(module, "lora_a"):
            module.lora_a = module.lora_a  # unfreeze by re-assigning
            module.lora_b = module.lora_b

    # ---- Load data ----
    from .auxdpo_data import load_dpo_data
    val_split = get("val_split", args.val_split, 0.1)
    max_seq = get("max_seq_length", args.max_seq_length, 2048)

    train_dataset, val_dataset = load_dpo_data(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_seq,
        val_split=val_split,
        seed=seed,
    )

    # ---- Set up trainer ----
    from .auxdpo_trainer import AuxDPOTrainer, AuxDPOArgs

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = AuxDPOArgs(
        batch_size=get("batch_size", args.batch_size, 1),
        iters=get("iters", args.iters, 500),
        learning_rate=get("learning_rate", args.lr, 5e-6),
        aux_lr=get("aux_lr", args.aux_lr, 5e-3),
        beta=get("beta", args.beta, 0.1),
        label_smoothing=get("label_smoothing", args.label_smoothing, 0.0),
        use_auxdpo=not args.no_aux and not config.get("no_aux", False),
        lambda_null=get("lambda_null", args.lambda_null, 1.0),
        lambda_reg=get("lambda_reg", args.lambda_reg, 0.01),
        delta_cap=get("delta_cap", args.delta_cap, 1.0),
        max_seq_length=max_seq,
        grad_checkpoint=args.grad_checkpoint or config.get("grad_checkpoint", False),
        steps_per_report=get("report_every", args.report_every, 10),
        steps_per_eval=get("eval_every", args.eval_every, 200),
        steps_per_save=get("save_every", args.save_every, 100),
        adapter_file=str(output_dir / "adapters.safetensors"),
        val_batches=get("val_batches", args.val_split, 25),
    )

    # Save adapter config (compatible with mlx_lm format)
    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": lora_params,
        "auxdpo": {
            "beta": training_args.beta,
            "lambda_null": training_args.lambda_null,
            "lambda_reg": training_args.lambda_reg,
            "delta_cap": training_args.delta_cap,
            "use_auxdpo": training_args.use_auxdpo,
        },
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    trainer = AuxDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
    )

    # ---- Train ----
    trainer.train()

    print()
    print("=" * 60)
    print(f"Training complete. Adapter saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
