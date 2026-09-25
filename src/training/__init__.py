"""BaFuse v2 training package."""
from .degradation_trainer import DegradationTrainer
from .multitask_trainer import MultiTaskTrainer

__all__ = ["DegradationTrainer", "MultiTaskTrainer"]
