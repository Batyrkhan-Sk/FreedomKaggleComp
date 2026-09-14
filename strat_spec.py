"""Do the spectral changes land on UNCERTAIN matches or break CONFIDENT ones?
A re-ranker that only perturbs coin-flips is cheap insurance; one that overrides
the descriptor where it is most sure is how you lose 0.04 in one submission."""
import numpy as np
from task1_concat import l2
d=np.load('eigs_k50_g3.npz'); lam,U=d['lam'][:2048],d['U'][:,:2048]
X=np.load('base5120.npy'); cand=np.load('rot_candidates.npy')
scale=1.0/np.sqrt(np.maximum(1.0-0.99*lam,1e-6))
phi=l2((U*scale[None,:]).astype(np.float32))

def top1(F,qs):
    o=np.zeros(len(qs),np.int64); s=np.zeros(len(qs),np.float32)
    for i in range(0,len(qs),256):
        b=qs[i:i+256]; S=(F[b]@F.T).astype(np.float32); S[np.arange(len(b)),b]=-2
        o[i:i+len(b)]=S.argmax(1); s[i:i+len(b)]=S.max(1)
    return o,s
b1,bs=top1(X,cand)
q=np.quantile(bs,[0.25,0.5,0.75])
print(f"base top-1 similarity: p25={q[0]:.3f} p50={q[1]:.3f} p75={q[2]:.3f}")
print(f"{'blend':>6}{'overall':>9}{'low-conf':>10}{'mid':>8}{'HIGH-conf':>11}")
for bl in (8.0,4.0,2.0,0.0):
    out = phi if bl<=0 else l2(np.hstack([phi,bl*l2(X)]).astype(np.float32))
    s1,_=top1(out,cand); ch=b1!=s1
    lo=bs<=q[0]; mid=(bs>q[0])&(bs<q[2]); hi=bs>=q[2]
    print(f"{bl:>6.1f}{ch.mean()*100:>8.1f}%{ch[lo].mean()*100:>9.1f}%"
          f"{ch[mid].mean()*100:>7.1f}%{ch[hi].mean()*100:>10.1f}%")
