"""DINOv2 on exactly the same gallery subset and the same query images as vlad_test."""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

Image.MAX_IMAGE_PIXELS=None
D = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
N_GALLERY, N_QUERY, SEED = 4000, 200, 1234
SIZE=518
MEAN,STD=(0.485,0.456,0.406),(0.229,0.224,0.225)

paths = sorted(D.glob('*.png'))[:N_GALLERY]
# reuse the ViT-L @518 features already on disk for these same images
feats = np.load('features_l518.npy')[:N_GALLERY]
names = np.load('features_l518_names.npy', allow_pickle=True)[:N_GALLERY]
assert list(names) == [p.name for p in paths], "feature order must match"
print(f'gallery {feats.shape}')

rng = np.random.default_rng(SEED)
picks = rng.choice(len(paths), size=N_QUERY, replace=False)

tf = transforms.Compose([transforms.Resize((SIZE,SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                         transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
dev='mps' if torch.backends.mps.is_available() else 'cpu'
model=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',verbose=False).eval().to(dev)
def gem(pt,p=3.,eps=1e-6): return pt.clamp(min=eps).pow(p).mean(1).pow(1/p)

# the exact query images vlad_test wrote
qfiles=[Path('/tmp/vq')/f'{k}.png' for k in range(N_QUERY)]
assert all(f.exists() for f in qfiles), "run vlad_test.py first"
out=[]
with torch.no_grad():
    for i in range(0,len(qfiles),4):
        b=torch.stack([tf(Image.open(f).convert('RGB')) for f in qfiles[i:i+4]]).to(dev)
        o=model.forward_features(b)
        v=torch.cat([F.normalize(o['x_norm_clstoken'],dim=1),
                     F.normalize(gem(o['x_norm_patchtokens']),dim=1)],dim=1)
        out.append(v.float().cpu())
Q=torch.cat(out).numpy()

def l2(x,eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)
def hits(q,g,picks,dim=0):
    q,g=l2(q.astype(np.float64)),l2(g.astype(np.float64))
    if dim:
        mu=g.mean(0,keepdims=True); _,s,vt=np.linalg.svd(g-mu,full_matrices=False)
        d=min(dim,g.shape[1]); sc=s[:d]/np.sqrt(len(g)-1)
        g=l2((g-mu)@vt[:d].T/(sc+1e-8)); q=l2((q-mu)@vt[:d].T/(sc+1e-8))
    o=np.argsort(-(q@g.T),axis=1)
    return (float(np.mean([picks[i]==o[i,0] for i in range(len(picks))])),
            float(np.mean([picks[i] in o[i,:3] for i in range(len(picks))])))

h1,h3=hits(Q,feats,picks)
print(f'DINOv2 ViT-L@518 raw          Hit@1 {h1:.3f}  Hit@3 {h3:.3f}')
h1w,h3w=hits(Q,feats,picks,dim=1536)
print(f'DINOv2 ViT-L@518 PCA-whitened Hit@1 {h1w:.3f}  Hit@3 {h3w:.3f}')
