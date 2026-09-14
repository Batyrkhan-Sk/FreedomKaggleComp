"""7B paired with the STRONGEST available partner, not the weakest.

7B alone at 8192 scored 0.95302, beating the 4-block 9728 (0.92617). So capacity
dominates diversity -- which means the partner block should be the best one we
have, not ViT-L@518 (0.79194 standalone, the weakest).

C: 7B + DINOv3-H+@768  -> 10752 dims (H+ is the second-strongest block)
D: 7B + ViT-g@518      -> 11264 dims (0.8285 standalone; nearer the 12800 OOM)

Both drop the rotation correction, which variant B keeps -- so B and C/D test
different trades: ~7 rotation queries versus a much stronger second block.
"""
import numpy as np, pandas as pd
from task1_concat import l2, whiten

names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
D7 = np.load('features_d7b512.npy').astype(np.float64)
HP = np.load('features_dinov3.npy').astype(np.float64)      # H+@768, 2560
G  = np.load('kout/lost-in-the-museum/features_g.npy').astype(np.float64)  # 3072
for nm, a in (('7B', D7), ('H+@768', HP), ('ViT-g', G)):
    assert np.isfinite(a).all(), f'{nm} non-finite'
    print(nm, a.shape, flush=True)

def emit(x, out, prec=5):
    print(f'  whitening {x.shape} ...', flush=True)
    f = whiten(x, x.shape[1], 1.0)
    df = pd.DataFrame(np.round(f, prec), columns=[f'feature_{i}' for i in range(f.shape[1])])
    df.insert(0, 'image_name', list(names)); df['ID'] = df['image_name']
    assert len(df) == 20000 and not df.isna().any().any()
    df.to_csv(out, index=False)
    print(f'  wrote {out}: {df.shape[0]} x {df.shape[1]}', flush=True)

emit(np.hstack([l2(D7), l2(HP)]), 'submission_7b_hp_10752.csv')   # C
emit(np.hstack([l2(D7), l2(G)]),  'submission_7b_g_11264.csv')    # D
