"""What fraction of real queries are rotated by a multiple of 90 degrees?

For each detected query, embed it at 0/90/180/270 and see which orientation
matches the gallery best. If a large share peak at something other than 0, that
share is currently unreachable and is the size of the available gain.
"""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from deskew import estimate_angle

Image.MAX_IMAGE_PIXELS=None
D=Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
MEAN,STD=(0.485,0.456,0.406),(0.229,0.224,0.225); SIZE=518
tf=transforms.Compose([transforms.Resize((SIZE,SIZE),interpolation=transforms.InterpolationMode.BICUBIC),
                       transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
dev='mps' if torch.backends.mps.is_available() else 'cpu'
model=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',verbose=False).eval().to(dev)
def gem(pt,p=3.,eps=1e-6): return pt.clamp(min=eps).pow(p).mean(1).pow(1/p)
def l2(x,eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)

G=l2(np.load('features_l518.npy').astype(np.float32))
names=list(np.load('features_l518_names.npy',allow_pickle=True))
idx={n:i for i,n in enumerate(names)}
paths=sorted(D.glob('*.png'))

rng=np.random.default_rng(11)
cand=[]
for i in rng.choice(len(paths),1400,replace=False):
    try: deg,_=estimate_angle(Image.open(paths[i]).convert('RGB'))
    except Exception: continue
    if deg!=0.0: cand.append(paths[int(i)])
    if len(cand)>=45: break
print(f'{len(cand)} detected queries', flush=True)

def embed(imgs):
    out=[]
    with torch.no_grad():
        for i in range(0,len(imgs),4):
            b=torch.stack([tf(im) for im in imgs[i:i+4]]).to(dev)
            o=model.forward_features(b)
            v=torch.cat([F.normalize(o['x_norm_clstoken'],dim=1),
                         F.normalize(gem(o['x_norm_patchtokens']),dim=1)],dim=1)
            out.append(v.float().cpu())
    return l2(torch.cat(out).numpy())

best_rot=[]
for p in cand:
    im=Image.open(p).convert('RGB')
    E=embed([im.rotate(a, expand=True) for a in (0,90,180,270)])
    self_i=idx[p.name]
    peaks=[]
    for k in range(4):
        s=E[k]@G.T; s[self_i]=-1
        peaks.append(float(s.max()))
    best_rot.append((p.name, int(np.argmax(peaks))*90, peaks))

from collections import Counter
c=Counter(r for _,r,_ in best_rot)
print('\nbest-matching orientation across detected queries:')
for r in (0,90,180,270):
    print(f'  {r:>3} deg : {c.get(r,0):2d}/{len(best_rot)}  ({100*c.get(r,0)/len(best_rot):.0f}%)')
gains=[max(p)-p[0] for _,r,p in best_rot if r!=0]
if gains:
    print(f'\nfor the non-upright ones, similarity gain from rotating: mean {np.mean(gains):.3f}, max {max(gains):.3f}')
