"""Does the deliverable notebook's combine logic reproduce the 0.86912 file?

The extraction cells are proven code -- they produced the .npy files on disk.
The genuinely new part is the combine (rotation flip -> per-block L2 -> concat ->
full-width whitening), so run exactly that against the existing features and
diff against the submission that actually scored 0.86912.

Extracts cell 3 from the notebook and executes it, so what is verified is the
notebook's own source rather than a copy that can drift from it.
"""
import base64, json, zlib

import numpy as np
import pandas as pd

nb = json.load(open('kaggle-museum-solution.ipynb'))
cell1 = ''.join(nb['cells'][1]['source'])
cell3 = ''.join(nb['cells'][3]['source'])

# the embedded candidate list, taken from the notebook itself
ns = {'np': np, 'base64': base64, 'zlib': zlib}
exec(cell1.split('class Imgs')[0], ns)
CAND = ns['CAND']
ref = np.load('rot_candidates.npy')
assert np.array_equal(CAND, ref), 'embedded candidates differ from rot_candidates.npy'
print(f'embedded candidate list matches rot_candidates.npy ({len(CAND)})')

G = np.load('kout/lost-in-the-museum/features_g.npy')
L0 = np.load('features_l518.npy')
L180 = np.load('features_l518_rot180.npy')
names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)


class _Work:
    def __truediv__(self, other):
        return '/tmp/nb_' + str(other)


g = {'G': G, 'L0': L0, 'L180': L180, 'CAND': CAND, 'names': names,
     'np': np, 'MARGIN': 0.15, 'WORK': _Work()}
exec(cell3, g)

a = pd.read_csv('/tmp/nb_submission.csv', dtype={'image_name': str, 'ID': str})
b = pd.read_csv('submission_gL_rot5120.csv', dtype={'image_name': str, 'ID': str})
print(f'\nnotebook  {a.shape}\nreference {b.shape}')
assert list(a.columns) == list(b.columns), 'column mismatch'
assert (a.image_name.values == b.image_name.values).all(), 'row order mismatch'
fa = a.drop(columns=['image_name', 'ID']).to_numpy(np.float32)
fb = b.drop(columns=['image_name', 'ID']).to_numpy(np.float32)
d = np.abs(fa - fb)
print(f'max abs diff {d.max():.2e}   mean {d.mean():.2e}')
cos = (fa * fb).sum(1) / (np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1) + 1e-12)
print(f'per-row cosine: min {cos.min():.6f}  mean {cos.mean():.6f}')
print('\nVERDICT:', 'EXACT MATCH' if d.max() < 1e-5 else
      ('equivalent (sign/rotation only)' if cos.min() > 0.9999 else 'MISMATCH'))
