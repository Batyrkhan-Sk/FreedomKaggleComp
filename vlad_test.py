"""Does VLAD beat DINOv2 on this task's retrieval?

Validation uses synthetic queries built with the corrected augmentation. That is
legitimate here in a way it was not for fine-tuning: SIFT and VLAD learn nothing
from the data, so there is no circularity -- the method cannot memorise an
augmentation it was never shown.

Run on a subset so it finishes in minutes; the comparison against DINOv2 uses
the same subset and the same synthetic queries.
"""
import sys, time
import numpy as np, torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, 'finetune')
from augment2 import query_view, IMAGENET_MEAN, IMAGENET_STD
from vlad import root_sift, encode_vlad, train_codebook

Image.MAX_IMAGE_PIXELS = None
D = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
N_GALLERY, N_QUERY, SEED = int(sys.argv[1]) if len(sys.argv)>1 else 4000, 200, 1234

paths = sorted(D.glob('*.png'))[:N_GALLERY]
print(f'gallery {len(paths)}')

t0=time.time()
cb = train_codebook(paths, n_images=800)
print(f'codebook {cb.shape} in {time.time()-t0:.0f}s', flush=True)

t0=time.time()
G = np.zeros((len(paths), cb.shape[0]*128), dtype=np.float32)
for i,p in enumerate(paths):
    G[i] = encode_vlad(root_sift(p), cb)
    if (i+1) % 500 == 0:
        r=(i+1)/(time.time()-t0); print(f'  {i+1}/{len(paths)} {r:.0f} img/s ETA {(len(paths)-i-1)/r/60:.1f}m', flush=True)
print(f'gallery VLAD in {(time.time()-t0)/60:.1f} min', flush=True)

rng = np.random.default_rng(SEED)
picks = rng.choice(len(paths), size=N_QUERY, replace=False)
tf = query_view(512, 1.0)
np.random.seed(SEED); torch.manual_seed(SEED)
qimgs=[]
for i in picks:
    t = tf(Image.open(paths[i]).convert('RGB'))
    a = (t.permute(1,2,0).numpy()*np.array(IMAGENET_STD)+np.array(IMAGENET_MEAN)).clip(0,1)
    qimgs.append(Image.fromarray((a*255).astype('uint8')))

tmp = Path('/tmp/vq'); tmp.mkdir(exist_ok=True)
Q = np.zeros((len(qimgs), cb.shape[0]*128), dtype=np.float32)
for k,im in enumerate(qimgs):
    f = tmp/f'{k}.png'; im.save(f)
    Q[k] = encode_vlad(root_sift(f), cb)

def l2(x, eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)
def hits(q, g, picks, dim=0):
    q, g = l2(q.astype(np.float64)), l2(g.astype(np.float64))
    if dim:
        mu=g.mean(0,keepdims=True); _,s,vt=np.linalg.svd(g-mu, full_matrices=False)
        d=min(dim,g.shape[1]); sc=s[:d]/np.sqrt(len(g)-1)
        g=l2((g-mu)@vt[:d].T/(sc+1e-8)); q=l2((q-mu)@vt[:d].T/(sc+1e-8))
    o=np.argsort(-(q@g.T),axis=1)
    return (float(np.mean([picks[i]==o[i,0] for i in range(len(picks))])),
            float(np.mean([picks[i] in o[i,:3] for i in range(len(picks))])))

h1,h3 = hits(Q,G,picks)
print(f'\nVLAD raw          Hit@1 {h1:.3f}  Hit@3 {h3:.3f}')
h1w,h3w = hits(Q,G,picks,dim=1536)
print(f'VLAD PCA-whitened Hit@1 {h1w:.3f}  Hit@3 {h3w:.3f}')
