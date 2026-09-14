"""Run the v2 notebook's OWN fusion code against the real feature files and diff
the result against submission_4block_9728.csv (the artifact that scored 0.92617).
A deliverable that does not reproduce its submission is worse than none."""
import json, numpy as np, pandas as pd

nb = json.load(open('kaggle-museum-solution-v2.ipynb'))
fuse = ''.join(nb['cells'][4]['source'])
ns = {'np': np}
exec(fuse.split('g  = l2(')[0], ns)          # take l2 + whiten exactly as shipped
l2, whiten = ns['l2'], ns['whiten']

D768 = np.load('features_dinov3.npy')
Lc   = np.load('l518_rotcorrected.npy')       # rotation flip already applied, margin 0.15
g    = np.load('kout/lost-in-the-museum/features_g.npy')
D512 = np.load('features_dinov3_512.npy')
x = np.hstack([l2(D768.astype(np.float64)), l2(Lc.astype(np.float64)),
               l2(g.astype(np.float64)), l2(D512.astype(np.float64))])
print('concatenated', x.shape, flush=True)
f = whiten(x, x.shape[1], 1.0)
print('whitened', f.shape, flush=True)

ref = pd.read_csv('submission_4block_9728.csv')
cols = [c for c in ref.columns if c.startswith('feature_')]
R = ref[cols].to_numpy(np.float32)
F = np.round(f, 5)
print('shapes', F.shape, R.shape)
print('max abs diff :', float(np.abs(F - R).max()))
print('mean cosine  :', float(np.mean((l2(F.astype(np.float64)) * l2(R.astype(np.float64))).sum(1))))
print('IDENTICAL    :', bool(np.allclose(F, R, atol=2e-5)))
