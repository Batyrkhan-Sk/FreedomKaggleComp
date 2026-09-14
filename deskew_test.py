"""Does removing the rotation make a query retrieve its original?

The gallery embeddings on disk were computed from the untouched images, which is
exactly the real setup: gallery scans are upright, queries are not. So we embed
each detected query twice -- as-is and deskewed -- and compare how strongly each
matches the rest of the corpus.

A query's own row is excluded, otherwise it trivially matches itself.
"""
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from deskew import estimate_angle, deskew

Image.MAX_IMAGE_PIXELS = None
D = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
MEAN, STD = (0.485,0.456,0.406), (0.229,0.224,0.225)
SIZE = 518

feats = np.load('features_l518.npy'); names = np.load('features_l518_names.npy', allow_pickle=True)
gal = (feats/np.linalg.norm(feats,axis=1,keepdims=True)).astype(np.float32)
idx_of = {n: i for i, n in enumerate(names)}

paths = sorted(D.glob('*.png'))
rng = np.random.default_rng(0)
sample = [paths[i] for i in rng.choice(len(paths), 2500, replace=False)]
print('scanning for rotated queries...', flush=True)
found = []
for p in sample:
    try:
        deg, gain = estimate_angle(Image.open(p).convert('RGB'))
    except Exception:
        continue
    if deg != 0.0 and gain > 0.12:
        found.append((p, deg, gain))
found.sort(key=lambda t: -t[2])
print(f'detected {len(found)} rotated candidates in {len(sample)} sampled images '
      f'({100*len(found)/len(sample):.1f}%)', flush=True)

tf = transforms.Compose([transforms.Resize((SIZE,SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                         transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',verbose=False).eval().to(dev)

def gem(pt,p=3.,eps=1e-6): return pt.clamp(min=eps).pow(p).mean(1).pow(1/p)
def embed(imgs):
    out=[]
    with torch.no_grad():
        for i in range(0,len(imgs),4):
            b=torch.stack([tf(im) for im in imgs[i:i+4]]).to(dev)
            o=model.forward_features(b)
            v=torch.cat([F.normalize(o['x_norm_clstoken'],dim=1),
                         F.normalize(gem(o['x_norm_patchtokens']),dim=1)],dim=1)
            out.append(v.float().cpu())
    x=torch.cat(out).numpy()
    return x/np.linalg.norm(x,axis=1,keepdims=True)

test = found[:40]
orig  = [Image.open(p).convert('RGB') for p,_,_ in test]
fixed = [deskew(im)[0] for im in orig]
print('embedding...', flush=True)
eo, ef = embed(orig), embed(fixed)

rows=[]
for k,(p,deg,gain) in enumerate(test):
    self_i = idx_of[p.name]
    so = eo[k] @ gal.T; so[self_i] = -1
    sf = ef[k] @ gal.T; sf[self_i] = -1
    rows.append((p.name, deg, float(so.max()), names[int(so.argmax())],
                 float(sf.max()), names[int(sf.argmax())]))

import statistics
b = [r[2] for r in rows]; a = [r[4] for r in rows]
changed = sum(1 for r in rows if r[3] != r[5])
print(f'\nn={len(rows)} detected queries')
print(f'  top-1 similarity BEFORE deskew: mean {statistics.mean(b):.4f}')
print(f'  top-1 similarity AFTER  deskew: mean {statistics.mean(a):.4f}')
print(f'  improved: {sum(1 for x,y in zip(b,a) if y>x)}/{len(rows)}   best match changed: {changed}')
print('\nlargest gains:')
for r in sorted(rows, key=lambda r: r[2]-r[4])[:8]:
    print(f'  {r[0]} ang={r[1]:+5.1f}  {r[2]:.3f}->{r[4]:.3f}   match {r[3]} -> {r[5]}')
