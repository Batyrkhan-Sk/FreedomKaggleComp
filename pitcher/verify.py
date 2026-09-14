"""Compare two audit runs without labels, and guard the five known answers.

There is no dev set, so a change is judged three ways: the five correct answers
shipped in submission.csv must stay correct, no question pattern may start
landing in a worse region of the JSON, and no leaf name may spike -- a spike is
what exposed `ip_1` (outs sold as innings) and `games.play` (a games-played
count sold as a fielding position).
"""
import argparse, collections, csv, re, sys


def load(p):
    return {r['ID']: r for r in csv.DictReader(open(p))}


def ground_truth():
    return {r['ID']: r['ANSWER'] for r in csv.DictReader(open('submission.csv'))
            if r['ANSWER'].strip().lower() != 'no answer'}


PATTERNS = {
    'position':     (r'\b(what|which) position\b|\bposition does\b', r'\.(primary_)?position$'),
    'primary pos':  (r'\bprimary position\b',                       r'\.primary_position$'),
    # "injured X" is often just a descriptor on a profile question
    'injury detail':(r'injur\w*\s+(date|status)|(update|start) date',  r'injuries'),
    'standings':    (r'games back|wild card|elimination|standings',  r'standings'),
    'recency game': (r'(matches|games) ago|previous match',          r'last_10_games'),
    # These two used to accept any path at all, which is how a month regression
    # scored 100%. The expected region has to be something a wrong answer can fail.
    # A question naming a full calendar date resolves through the game anchor,
    # whose path carries no month segment, so it is not a month-split question.
    'month split':  (r'\b(january|february|march|april|may|june|july|august|september|october)\b(?!\s+\d{1,2},)',
                     r'(january|february|march|april|may|june|july|august|september|october|last_10_games)'),
    'day/night':    (r'\b(day|night) games?\b',                      r'day_night'),
    'against team': (r'\bagainst the\b',                             r'.'),
    'own stat':     (r'\b(matches|games) ago\b',                     r'\.players\.', 'player'),
    'affiliate':    (r'affiliate',                                   r'minorleague_affiliate'),
    'salary':       (r'\bsalary\b',                                  r'\.salary$'),
    'jersey':       (r'\bjersey\b',                                  r'jersey'),
}


def report(audit, label):
    rows = list(audit.values())
    print(f'\n=== {label} ===')
    gt = ground_truth()
    ok = sum(1 for k, v in gt.items() if audit[k]['answer'] == v)
    print(f'known-correct answers held: {ok}/{len(gt)}' + ('' if ok == len(gt) else '   <-- REGRESSION'))
    for k, v in gt.items():
        if audit[k]['answer'] != v:
            print(f'   LOST {k}: got {audit[k]["answer"]!r} want {v!r}')

    print('\nquestion pattern -> expected region:')
    for name, spec in PATTERNS.items():
        qre, pre = spec[0], spec[1]
        src_filter = spec[2] if len(spec) > 2 else None
        hit = [r for r in rows if re.search(qre, r['question'], re.I)
               and (src_filter is None or r['source'] == src_filter)]
        if not hit:
            continue
        good = sum(1 for r in hit if re.search(pre, r['path']))
        print(f'  {name:14s} {good:3d}/{len(hit):-3d}  {good/len(hit):5.0%}')

    print('\nleaf frequency (spikes are suspicious):')
    leaf = collections.Counter(r['path'].split('.')[-1] for r in rows if r['path'] != '-')
    for k, v in leaf.most_common(8):
        print(f'  {v:3d}  {k}')
    na = sum(1 for r in rows if r['answer'].strip().lower() == 'no answer')
    print(f'\nno-answer rows: {na}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('new'); ap.add_argument('old', nargs='?')
    a = ap.parse_args()
    new = load(a.new)
    report(new, a.new)
    if a.old:
        old = load(a.old)
        changed = [k for k in new if new[k]['answer'] != old[k]['answer']]
        print(f'\n=== {len(changed)} answers changed vs {a.old} ===')
        for k in changed:
            print(f'  {new[k]["question"][:74]}')
            print(f'    -  {old[k]["path"][-56:]} = {old[k]["answer"][:22]}')
            print(f'    +  {new[k]["path"][-56:]} = {new[k]["answer"][:22]}')


if __name__ == '__main__':
    main()
