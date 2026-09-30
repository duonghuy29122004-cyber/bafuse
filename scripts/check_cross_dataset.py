"""Quick check: EIS adapter forward pass for all 14 models."""
import sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src"))

from src.models.benchmark_models import get_model
from scripts.cross_dataset_benchmark import EISOnlyModel, ALL_MODELS

ok, failed = [], []
for name in ALL_MODELS:
    try:
        arch  = get_model(name, 3, 3, 4)
        model = EISOnlyModel(arch, mendeley_eis_dim=8)
        n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x     = torch.randn(4, 8)
        lli, lam, cl = model(x)
        assert lli.shape == (4,1) and lam.shape == (4,1) and cl.shape == (4,1)
        ok.append(name)
        print(f"  [PASS] {name:4s}  adapter_params={n:,}  out={tuple(lli.shape)}")
    except Exception as e:
        failed.append(name)
        print(f"  [FAIL] {name:4s}  {e}")

print(f"\n{len(ok)}/14 PASS   FAILED: {failed or 'none'}")
