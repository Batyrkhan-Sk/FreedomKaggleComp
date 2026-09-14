"""VLAD over the full corpus, plus submissions alone and fused with DINOv2."""
import time
import numpy as np, pandas as pd
from pathlib import Path
from vlad import root_sift, encode_vlad, train_codebook

D = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
paths = sorted(D.glob('*.png'))
print(f'{len(paths)} images')

t0=time.time(); cb = train_codebook(paths, n_images=1500)
print(f'codebook {cb.shape} in {(time.time()-t0)/60:.1f} min', flush=True)

# Checkpoint every 2000 images: this run has already been killed once at
# 14,000/20,000 by a parent-shell timeout, and re-doing 8 minutes of SIFT for
# nothing is avoidable.
t0=time.time()
CKPT, CKPT_DONE = 'vlad_ckpt.npy', 'vlad_ckpt_done.npy'
V = np.zeros((len(paths), cb.shape[0]*128), dtype=np.float32)
done = np.zeros(len(paths), dtype=bool)
if Path(CKPT).exists() and Path(CKPT_DONE).exists():
    cv, cd = np.load(CKPT), np.load(CKPT_DONE)
    if cv.shape == V.shape:
        V, done = cv, cd
        print(f'resuming: {done.sum()}/{len(paths)} already encoded', flush=True)

for i,p in enumerate(paths):
    if done[i]:
        continue
    V[i] = encode_vlad(root_sift(p), cb)
    done[i] = True
    if (i+1) % 2000 == 0:
        r=(done.sum())/(time.time()-t0)
        print(f'  {i+1}/{len(paths)}  {r:.0f} img/s  ETA {(len(paths)-i-1)/max(r,1e-6)/60:.1f} min', flush=True)
        np.save(CKPT, V); np.save(CKPT_DONE, done)
assert done.all(), f'only {done.sum()}/{len(paths)} encoded'
np.save('features_vlad.npy', V)
for f in (CKPT, CKPT_DONE):
    Path(f).unlink(missing_ok=True)
print(f'VLAD {V.shape} in {(time.time()-t0)/60:.1f} min', flush=True)

def l2(x,eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)
def whiten_to(x, dim):
    x=l2(x.astype(np.float64)); mu=x.mean(0,keepdims=True)
    _,s,vt=np.linalg.svd(x-mu,full_matrices=False)
    d=min(dim,x.shape[1]); sc=s[:d]/np.sqrt(len(x)-1)
    print(f'  PCA {x.shape[1]} -> {d}  explains {(s[:d]**2).sum()/(s**2).sum():.1%}')
    return l2((x-mu)@vt[:d].T/(sc+1e-8)).astype(np.float32)

names=[p.name for p in paths]
def write(x, fn):
    df=pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
    df.insert(0,'image_name',names); df['ID']=df['image_name']
    df.to_csv(fn, index=False, float_format='%.6f')
    print(f'wrote {fn}: {len(df)} rows, {df.shape[1]} cols')

write(whiten_to(V, 1536), 'submission_vlad.csv')

dino = np.load('features_l518.npy')
fused = np.concatenate([l2(dino.astype(np.float64)), l2(V.astype(np.float64))], axis=1)
write(whiten_to(fused, 1536), 'submission_fused.csv')
