"""Does rotation-averaging fix upside-down queries?

Query 02200 was confirmed by eye to be its top-1 match rotated 180 degrees, and
DINOv2 scored that pair at only 0.280 because it has no invariance to large
rotations. If that is the general failure, averaging each descriptor over
0/90/180/270 degrees should raise such pairs sharply.

Unlike every earlier offline test on this task, this one has real ground truth:
pairs verified visually, not synthesised.
"""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

Image.MAX_IMAGE_PIXELS=None
D=Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
MEAN,STD=(0.485,0.456,0.406),(0.229,0.224,0.225); SIZE=518
tf=transforms.Compose([transforms.Resize((SIZE,SIZE),interpolation=transforms.InterpolationMode.BICUBIC),
                       transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
dev='mps' if torch.backends.mps.is_available() else 'cpu'
model=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',verbose=False).eval().to(dev)
def gem(pt,p=3.,eps=1e-6): return pt.clamp(min=eps).pow(p).mean(1).pow(1/p)

def embed_imgs(imgs):
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

def plain(name):
    return embed_imgs([Image.open(D/name).convert('RGB')])[0]

def rot_avg(name):
    im=Image.open(D/name).convert('RGB')
    views=[im.rotate(a, expand=True) for a in (0,90,180,270)]
    v=embed_imgs(views).mean(0)
    return v/np.linalg.norm(v)

# pairs confirmed by eye: query <-> its true match
PAIRS=[('02200.png','?'),('06572.png','?')]
# resolve each query's current top-1 from the ViT-g gallery for reference
feats=np.load('kout/lost-in-the-museum/features_g.npy')
names=list(np.load('kout/lost-in-the-museum/feature_names.npy',allow_pickle=True))
def l2(x,eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)
X=l2(feats.astype(np.float64))
mu=X.mean(0,keepdims=True); _,s,vt=np.linalg.svd(X-mu,full_matrices=False)
d=1536; sc=s[:d]/np.sqrt(len(X)-1); G=l2((X-mu)@vt[:d].T/(sc+1e-8)).astype(np.float32)

resolved=[]
for q,_ in PAIRS:
    qi=names.index(q); sims=G[qi]@G.T; sims[qi]=-1
    j=int(sims.argmax()); resolved.append((q,names[j]))
    print(f'{q} -> top1 {names[j]}  (whitened sim {sims[j]:.3f})')

print()
for q,m in resolved:
    pq,pm=plain(q),plain(m)
    rq,rm=rot_avg(q),rot_avg(m)
    print(f'{q} <-> {m}')
    print(f'   plain cosine        {float(pq@pm):.4f}')
    print(f'   rotation-averaged   {float(rq@rm):.4f}')
