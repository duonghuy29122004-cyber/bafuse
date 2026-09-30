import sys; sys.path.insert(0,'d:/Capstone project/bafuse'); sys.path.insert(0,'d:/Capstone project/bafuse/src')
from src.models.benchmark_models import all_model_param_counts, get_model
import torch

counts = all_model_param_counts()
print('=== Extended model parameter counts ===')
for n in ['C','C2','D','D2','E','E3','E2','E4','E5','G','G2']:
    tag = ' <-- NEW' if n in ('C2','D2','E3','E4','E5','G2') else ''
    print(f'  {n:4s}: {counts[n]:>10,}{tag}')

print()
print('=== Forward pass check ===')
ok = True
for n in ['C2','D2','E3','E4','E5','G2']:
    try:
        m = get_model(n)
        d = torch.randn(4, 100, 3); e = torch.randn(4,3); p = torch.randn(4,4)
        out = m(d,e,p)
        shape = tuple(out['soh_pred'].shape)
        assert shape == (4,1), f'wrong shape {shape}'
        print(f'  {n}: PASS  shape={shape}')
    except Exception as ex:
        print(f'  {n}: FAIL  {ex}')
        ok = False

print()
print('All OK' if ok else 'SOME FAILED')
