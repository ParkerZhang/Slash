# Weighting New Tokens — Financial Identifier Family

## Concept

4 "vacuum" dimensions in the 384-dim embedding space have near-zero activation for all 30,522 existing tokens. These create a subspace where new financial identifier tokens can be placed, fully separated from existing tokens without any training.

## The 4 Vacuum Dimensions

| Dim | Impact | Mean  | Max\|val\| | Assigned Token |
|-----|--------|-------|-----------|----------------|
| 18  | 11.90  | 0.002 | 0.25      | `#MIC`         |
| 62  | 11.59  | -0.003| 0.51      | `ISIN`         |
| 28  | 11.69  | -0.007| 0.25      | `SEDOL`        |
| 245 | 11.77  | -0.002| 0.24      | `TICKER`       |

**Subspace properties:**
- Cross-correlation between dims: ≤ 0.17 (independent)
- Max existing projection onto unit vector: 0.39
- Existing tokens with projection > 4: **0 / 30,522**

## Embedding Formula (Shared Family Base)

All 4 tokens share a common base across all 4 vacuum dims, plus a boost in their assigned marker dim.

```
BASE  = 0.15   # shared across all 4 dims for all tokens
MARKER = 0.30  # additional boost in assigned dim

#MIC  → dims [18, 62, 28, 245] = [0.30, 0.15, 0.15, 0.15]
ISIN  → dims [18, 62, 28, 245] = [0.15, 0.30, 0.15, 0.15]
SEDOL → dims [18, 62, 28, 245] = [0.15, 0.15, 0.30, 0.15]
TICKER→ dims [18, 62, 28, 245] = [0.15, 0.15, 0.15, 0.30]
```

All other 380 dims = 0.

## Result

- Family cos similarity: **0.857** for all pairs
- Nearest existing token: **0.16–0.18** (5× farther than family)
- Clean NN ordering: self → family → existing tokens

## Repeat on New Machine

```python
import numpy as np, json, onnx
from onnx import numpy_helper

# 1. Load tokenizer and ONNX
vocab = json.load(open('tokenizer.json'))['model']['vocab']
model = onnx.load('model.onnx')

for init in model.graph.initializer:
    if init.name == 'embeddings.word_embeddings.weight':
        W = numpy_helper.to_array(init).copy()
        break

# 2. Token-to-Dim mapping
dims = {'#MIC': 18, 'ISIN': 62, 'SEDOL': 28, 'TICKER': 245}
BASE, MARKER = 0.15, 0.30

for name, tid in [(k, int(vocab[k])) for k in dims]:
    W[tid, :] = 0.0                              # zero all dims
    for d in [18, 62, 28, 245]:                  # set all 4 vacuum dims
        W[tid, d] = BASE
    W[tid, dims[name]] = MARKER                  # boost own marker dim

# 3. Save
new_init = numpy_helper.from_array(W, name=init.name)
model.graph.initializer[idx].CopyFrom(new_init)
onnx.save(model, 'model.onnx')
```

## Adding More Token Families (Future)

For a new family of K tokens, find K new vacuum dimensions (lowest downstream impact via `np.linalg.norm(W, axis=0)` on all weight matrices). Apply the same BASE/MARKER formula with those K dims.
