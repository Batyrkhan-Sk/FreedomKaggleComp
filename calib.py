"""Calibrate the synthetic queries so DINOv2 reproduces its real leaderboard score.

Without this the comparison is meaningless: at strength 1.0 DINOv2 scores 1.000
on a 4,000 gallery while its actual score is 0.792 on 20,000. A benchmark that
everything passes cannot rank methods.

So: full 20,000 gallery, and sweep the degradation strength until DINOv2
ViT-L@518 lands near 0.792. Whatever strength does that is the operating point
at which other methods should be compared.
"""
import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0,'finetune')
from augment2 import query_view, IMAGENET_MEAN, IMAGENET_STD

Image.MAX_IMAGE_PIXELS=None
D=Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
SIZE, NQ, SEED = 518, 250, 7
feats=np.load('features_l518.npy'); names=np.load('features_l518_names.npy',allow_pickle=True)
print(f'full gallery {feats.shape}')

tf=transforms.Compose([transforms.Resize((SIZE,SIZE),interpolation=transforms.InterpolationMode.BICUBIC),
                       transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
dev='mps' if torch.backends.mps.is_available() else 'cpu'
model=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',verbose=False).eval().to(dev)
def gem(pt,p=3.,eps=1e-6): return pt.clamp(min=eps).pow(p).mean(1).pow(1/p)
def l2(x,eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)

g=l2(feats.astype(np.float64))
mu=g.mean(0,keepdims=True); _,s,vt=np.linalg.svd(g-mu,full_matrices=False)
d=1536; sc=s[:d]/np.sqrt(len(g)-1)
gw=l2((g-mu)@vt[:d].T/(sc+1e-8))

rng=np.random.default_rng(SEED); picks=rng.choice(len(names),size=NQ,replace=False)
out_dir=Path('/tmp/calibq'); out_dir.mkdir(exist_ok=True)

for strength in (1.0, 1.6, 2.2, 2.8):
    qt=query_view(SIZE, strength)
    np.random.seed(SEED); torch.manual_seed(SEED)
    tens=[]
    for k,i in enumerate(picks):
        t=qt(Image.open(D/names[i]).convert('RGB'))
        a=(t.permute(1,2,0).numpy()*np.array(IMAGENET_STD)+np.array(IMAGENET_MEAN)).clip(0,1)
        Image.fromarray((a*255).astype('uint8')).save(out_dir/f's{strength}_{k}.png')
        tens.append(t)
    outs=[]
    with torch.no_grad():
        for i in range(0,len(tens),4):
            b=torch.stack(tens[i:i+4]).to(dev)
            o=model.forward_features(b)
            v=torch.cat([F.normalize(o['x_norm_clstoken'],dim=1),
                         F.normalize(gem(o['x_norm_patchtokens']),dim=1)],dim=1)
            outs.append(v.float().cpu())
    q=l2(torch.cat(outs).numpy().astype(np.float64))
    qw=l2((q-mu)@vt[:d].T/(sc+1e-8))
    o=np.argsort(-(qw@gw.T),axis=1)
    h3=float(np.mean([picks[i] in o[i,:3] for i in range(len(picks))]))
    print(f'  strength {strength:4.1f} -> DINOv2 Hit@3 {h3:.3f}   (real score is 0.792)', flush=True)
