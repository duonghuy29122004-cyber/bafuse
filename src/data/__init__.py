"""BaFuse data pipeline."""

from .parse_mat import load_mat_file, parse_discharge_curve, parse_eis_spectrum
from .pairing import pair_discharge_eis
from .split import split_by_battery

# dataset.py requires torch — import lazily to avoid DLL issues when torch unavailable
def _lazy_dataset():
    from .dataset import BaFuseDataset, create_dataloaders
    return BaFuseDataset, create_dataloaders

__all__ = [
    "load_mat_file",
    "parse_discharge_curve",
    "parse_eis_spectrum",
    "pair_discharge_eis",
    "split_by_battery",
]
