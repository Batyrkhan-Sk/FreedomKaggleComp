"""Fuse complementary descriptors into one submission.

VLAD alone scores 0.621 and ViT-L alone 0.792, but concatenating them reached
0.802 -- so local geometry carries information the semantic embedding lacks.
This applies the same trick to the stronger ViT-g backbone (0.8285), and to a
three-way combination.

Each descriptor is L2-normalised before concatenation so that neither dominates
by sheer magnitude, with an optional weight for the weaker one; then the usual
whitened PCA down to 1536 dims.
"""
import numpy as np, pandas as pd
from pathlib import Path

def l2(x, eps=1e-12): return x/(np.linalg.norm(x,axis=1,keepdims=True)+eps)

def whiten_to(x, dim=1536):
    x = l2(x.astype(np.float64)); mu = x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x-mu, full_matrices=False)
    d = min(dim, x.shape[1]); sc = s[:d]/np.sqrt(len(x)-1)
    print(f'    PCA {x.shape[1]} -> {d}  explains {(s[:d]**2).sum()/(s**2).sum():.1%}')
    return l2((x-mu)@vt[:d].T/(sc+1e-8)).astype(np.float32)

g    = np.load('kout/lost-in-the-museum/features_g.npy')       # ViT-g @518   0.8285
gnames = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
l    = np.load('features_l518.npy')                            # ViT-L @518   0.79194
v    = np.load('features_vlad.npy')                            # VLAD         0.62080
lnames = np.load('features_l518_names.npy', allow_pickle=True)

assert list(gnames) == list(lnames), 'row order must match across feature files'
print(f'ViT-g {g.shape}   ViT-L {l.shape}   VLAD {v.shape}   order verified')

names = list(gnames)
def write(x, fn):
    df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
    df.insert(0, 'image_name', names); df['ID'] = df['image_name']
    df.to_csv(fn, index=False, float_format='%.6f')
    print(f'  wrote {fn}: {len(df)} rows, {df.shape[1]} cols')

gL, lL, vL = l2(g.astype(np.float64)), l2(l.astype(np.float64)), l2(v.astype(np.float64))

print('A  ViT-g + VLAD'); write(whiten_to(np.concatenate([gL, vL], 1)), 'sub_g_vlad.csv')
print('B  ViT-g + 0.7*VLAD'); write(whiten_to(np.concatenate([gL, 0.7*vL], 1)), 'sub_g_vlad07.csv')
print('C  ViT-g + ViT-L + VLAD'); write(whiten_to(np.concatenate([gL, lL, vL], 1)), 'sub_g_l_vlad.csv')
