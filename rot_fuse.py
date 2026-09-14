"""Average the 0-degree and 180-degree embeddings into one descriptor.

44% of queries are rotated by a multiple of 90 degrees, and 180 is the largest
single group at 29%. Averaging the two orientations makes an upside-down query
and its upright gallery original produce the same vector by construction --
both sides receive identical treatment.

The remaining 90/270 passes would add ~15% more coverage; this tests the
mechanism first, on the group that matters most.
"""
import numpy as np, pandas as pd

def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

a = np.load('features_l518.npy')          # 0 degrees   (LB 0.79194 on its own)
b = np.load('features_l518_rot180.npy')   # 180 degrees
names = np.load('features_l518_names.npy', allow_pickle=True)
print(f'0deg {a.shape}   180deg {b.shape}')

# normalise each view BEFORE averaging, else whichever has larger magnitude wins
avg = l2(l2(a.astype(np.float64)) + l2(b.astype(np.float64)))

mu = avg.mean(axis=0, keepdims=True)
_, s, vt = np.linalg.svd(avg - mu, full_matrices=False)
d = 1536
x = l2((avg - mu) @ vt[:d].T / (s[:d] / np.sqrt(len(avg) - 1) + 1e-8)).astype(np.float32)
print(f'PCA {avg.shape[1]} -> {d}  explains {(s[:d]**2).sum()/(s**2).sum():.1%}')

df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
df.insert(0, 'image_name', list(names))
df['ID'] = df['image_name']
df.to_csv('submission_rot180.csv', index=False, float_format='%.6f')
print(f'wrote submission_rot180.csv: {len(df)} rows, {df.shape[1]} cols')
