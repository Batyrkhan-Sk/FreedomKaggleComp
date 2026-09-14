"""Run the deliverable notebook's OWN fusion code against the real 7B features
and diff against submission_7b_8192.csv -- the artifact that scored 0.95302."""
import json, numpy as np, pandas as pd
nb = json.load(open('kaggle-museum-solution-7b.ipynb'))
fuse = ''.join(nb['cells'][-1]['source'])
ns = {'np': np}
exec(fuse.split('assert np.isfinite(feats)')[0].replace('import pandas as pd',''), ns)
l2, whiten = ns['l2'], ns['whiten']

feats = np.load('features_d7b512.npy')
f = whiten(l2(feats.astype(np.float64)), feats.shape[1], 1.0)
ref = pd.read_csv('submission_7b_8192.csv')
R = ref[[c for c in ref.columns if c.startswith('feature_')]].to_numpy(np.float32)
F = np.round(f, 5)
print('shapes', F.shape, R.shape)
print('max abs diff :', float(np.abs(F - R).max()))
print('mean cosine  :', float(np.mean((l2(F.astype(np.float64))*l2(R.astype(np.float64))).sum(1))))
print('IDENTICAL    :', bool(np.allclose(F, R, atol=2e-5)))
