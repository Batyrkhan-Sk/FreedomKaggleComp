"""Two 7B fusions, built directly so no dead columns sneak in.

A: 7B alone, 8192 dims -- inside the width band that has actually worked.
   Three submissions above 9728 dims have now failed to beat the 9728 one
   (ConvNeXt swap and spectral both 0.92281 at 10752; 12800 OOMed), so width
   itself is suspect.
B: 7B + rotation-corrected ViT-L, 10240 dims -- the original plan. Keeps the
   orientation fix (~7 queries) at the cost of entering that suspect band.
"""
import numpy as np, pandas as pd
from task1_concat import l2, whiten

names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
D7 = np.load('features_d7b512.npy').astype(np.float64)
Lc = np.load('l518_rotcorrected.npy').astype(np.float64)
for nm, a in (('7B', D7), ('rot-ViT-L', Lc)):
    assert np.isfinite(a).all(), f'{nm} non-finite'
print('7B', D7.shape, '| rot-ViT-L', Lc.shape, flush=True)

def emit(x, out, prec=5):
    print(f'  whitening {x.shape} ...', flush=True)
    f = whiten(x, x.shape[1], 1.0)
    df = pd.DataFrame(np.round(f, prec), columns=[f'feature_{i}' for i in range(f.shape[1])])
    df.insert(0, 'image_name', list(names)); df['ID'] = df['image_name']
    assert len(df) == 20000 and not df.isna().any().any()
    df.to_csv(out, index=False)
    print(f'  wrote {out}: {df.shape[0]} x {df.shape[1]}', flush=True)

emit(l2(D7), 'submission_7b_8192.csv')                       # A
emit(np.hstack([l2(D7), l2(Lc)]), 'submission_7b_L_10240.csv')  # B
