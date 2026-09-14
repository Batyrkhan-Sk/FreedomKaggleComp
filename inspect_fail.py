"""What does the best model actually retrieve for a real query?

Every experiment so far has been blind: no ground truth, so no way to see *why*
a query fails. But real queries are identifiable by their rotation padding, and
the top matches can simply be looked at. If the top-1 is visibly the same
painting, the model is working and the failures are elsewhere. If it is a
different painting, we get to see exactly what confuses it.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from deskew import estimate_angle

Image.MAX_IMAGE_PIXELS = None
D = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
OUT = Path('/private/tmp/claude-501/-Users-a123-Desktop-KaggleComp/2b36c8ef-9ee6-47bd-bd68-eaed3fb9e981/scratchpad/fails')
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob('*.jpg'): f.unlink()

feats = np.load('kout/lost-in-the-museum/features_g.npy')          # ViT-g, 0.8285
names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)

def l2(x, eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)
x = l2(feats.astype(np.float64))
mu = x.mean(0, keepdims=True); _, s, vt = np.linalg.svd(x-mu, full_matrices=False)
d = 1536; sc = s[:d]/np.sqrt(len(x)-1)
G = l2((x-mu)@vt[:d].T/(sc+1e-8)).astype(np.float32)
print(f'gallery {G.shape}')

paths = sorted(D.glob('*.png'))
rng = np.random.default_rng(3)
sample = rng.choice(len(paths), 900, replace=False)
queries = []
for i in sample:
    try:
        deg, gain = estimate_angle(Image.open(paths[i]).convert('RGB'))
    except Exception:
        continue
    if deg != 0.0:
        queries.append(int(i))
    if len(queries) >= 8:
        break
print(f'{len(queries)} real queries found')

for qi in queries:
    sims = G[qi] @ G.T
    sims[qi] = -1
    top = np.argsort(-sims)[:3]
    tag = names[qi].replace('.png','')
    im = Image.open(D/names[qi]).convert('RGB'); im.thumbnail((260,260))
    im.save(OUT/f'{tag}_query.jpg')
    for r, j in enumerate(top, 1):
        m = Image.open(D/names[j]).convert('RGB'); m.thumbnail((260,260))
        m.save(OUT/f'{tag}_top{r}_{sims[j]:.3f}.jpg')
    print(f'{names[qi]}  top3 sims: ' + ', '.join(f'{sims[j]:.3f}' for j in top))
