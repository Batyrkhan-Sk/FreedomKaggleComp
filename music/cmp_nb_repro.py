import csv, sys
def load(p):
    d = {}
    for r in csv.DictReader(open(p)): d.setdefault(r['user_id'], []).append(r['item_id'])
    return d
a = load('submission_final.csv'); b = load('nb_repro.csv')
assert set(a) == set(b), 'user set mismatch'
ks = list(a)
ident = sum(a[u] == b[u] for u in ks)
ov = sum(len(set(a[u]) & set(b[u])) for u in ks) / sum(len(a[u]) for u in ks)
print(f'users {len(ks)}  byte-identical rankings: {ident}/{len(ks)}  slot overlap {ov:.6f}')
print('EXACT REPRODUCTION:', ident == len(ks))
