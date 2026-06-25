# AuxDPO: Auxiliary Direct Preference Optimization for MLX
# First known implementation on Apple Silicon / MLX framework
# Reference: arXiv:2510.20413

from .auxdpo_loss import auxdpo_loss, dpo_loss, get_sequence_logps
from .auxdpo_trainer import AuxDPOTrainer
from .auxdpo_data import DPODataset, load_dpo_data

__version__ = "0.1.0"
