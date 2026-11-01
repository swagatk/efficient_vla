"""Step 5b row checks: arms complete and paired, training rows and epochs, provenance across the resume (argument: run folder)."""
import collections, csv, glob, json, sys
D = sys.argv[1]
rd = lambda f: list(csv.DictReader(open(f)))
key = lambda r: (int(r['seed']), int(r['init_state']))
print('arms')
arms = {}
for who in ('shared', 'seed0', 'seed1', 'seed2'):
    for f in sorted(glob.glob(f'{D}/eval/{who}/part*/per_episode_*.csv')):
        arm = f.rsplit('per_episode_', 1)[1][:-4]
        arms.setdefault((who, arm), []).extend(rd(f))
for k, rows in arms.items():
    ks = [key(r) for r in rows]
    print(' ', k, len(rows), 'distinct', len(set(ks)), 'seeds', min(x[0] for x in ks), max(x[0] for x in ks), 'states', len({x[1] for x in ks}), 'task', {r['task_id'] for r in rows}, 'successes', sum(int(r['success']) for r in rows))
print('training')
for t in (0, 1, 2):
    rows = rd(f'{D}/train/seed{t}/train_log.csv'); up = rd(f'{D}/train/seed{t}/update_log.csv')
    c = collections.Counter(int(r['epoch']) for r in rows)
    pairs = [(int(r['epoch']), int(r['init_state']), int(r['seed'])) for r in rows]
    print(f'  seed{t}: rows', len(rows), 'distinct', len(set(pairs)), 'epochs', len(c), 'per epoch', set(c.values()), 'states', len({p[1] for p in pairs}), 'update rows', len(up), 'distinct', len({r['epoch'] for r in up}))
    rate = [sum(int(r['success']) for r in rows if int(r['epoch']) == e) / c[e] for e in sorted(c)]
    print('    first 5', round(sum(rate[:5]) / 5, 3), 'middle 5', round(sum(rate[16:21]) / 5, 3), 'last 5', round(sum(rate[-5:]) / 5, 3))
    print('    last updates', [(r['epoch'], r.get('weight_mean'), r.get('mean_abs'), r.get('std_mean')) for r in up[-1:]], 'first', [(r.get('weight_mean'), r.get('mean_abs'), r.get('std_mean')) for r in up[:1]])
print('provenance')
c = collections.Counter()
for f in glob.glob(f'{D}/**/provenance.jsonl', recursive=True):
    if '/pilot/' in f: continue
    for l in open(f):
        r = json.loads(l); st = r['settings']
        c[(r['git_commit'][:7], r['git_dirty'], tuple(r['gpus']), st.get('policy_path', '')[-14:], st.get('n_action_steps'), r['date'][:10])] += 1
for k, v in sorted(c.items(), key=str): print(' ', v, k)
import re
from math import comb
print('seam: seed 2 epoch 23, log attempts against train_log.csv')
log = {(int(r['epoch']), int(r['init_state'])): (int(r['success']), int(r['steps'])) for r in rd(f'{D}/train/seed2/train_log.csv')}
for k in range(4):
    text = open(f'{D}/logs/seed2_e023_part{k}.log').read()
    runs = [[(int(a), int(b), int(c)) for a, b, c in re.findall(r'epoch 23 init (\d+): success=(\d) steps=(\d+)', s)] for s in text.split('Traceback')]
    for i, run in enumerate(runs):
        print(f'  part{k} attempt {i}: {len(run)} episodes, match train_log {sum(log[(23, a)] == (b, c) for a, b, c in run)}')
print('per state, of 16 (systematic 2, 5, 6, 7, 8)')
names = [('base', 'shared', 'base'), ('prior', 'shared', 'prior')] + [(f'actor{t}', f'seed{t}', 'actor') for t in range(3)]
S_ = {}
for n, w, a in names:
    S_[n] = {key(r): int(r['success']) for r in arms[(w, a)]}
    print(f'  {n:7s}', [sum(v for k, v in S_[n].items() if k[1] == i) for i in range(10)])
SYS = {2, 5, 6, 7, 8}
def mcn(a, b, st):
    up = sum(a[k] > b[k] for k in a if k[1] in st); dn = sum(a[k] < b[k] for k in a if k[1] in st)
    n = up + dn; return up - dn, (min(1.0, 2 * sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n) if n else 1.0)
print('recomputed nets: net, p against base and prior, systematic and reachable')
for t in range(3):
    a = S_[f'actor{t}']; print(f'  actor{t}', 'sys/base', mcn(a, S_['base'], SYS), 'sys/prior', mcn(a, S_['prior'], SYS), 'reach/base', mcn(a, S_['base'], set(range(10)) - SYS), 'reach/prior', mcn(a, S_['prior'], set(range(10)) - SYS))
print('training rule: last 5 epochs minus epoch 0 of 3 seeds')
tr = {t: rd(f'{D}/train/seed{t}/train_log.csv') for t in range(3)}
e0 = [int(r['success']) for t in tr for r in tr[t] if int(r['epoch']) == 0]
print('  untrained', sum(e0), len(e0), round(sum(e0) / len(e0), 4))
for t in tr:
    l = [int(r['success']) for r in tr[t] if int(r['epoch']) >= 32]; print(f'  seed{t}', sum(l), len(l), round(sum(l) / len(l), 4), 'gain', round(100 * (sum(l) / len(l) - sum(e0) / len(e0)), 1), 'points')
ref = {key(r): int(r['success']) for r in rd(sys.argv[2])}
print('base against step 4 on the same episodes:', sum(ref.values()), sum(S_['base'].values()), 'differ', sum(ref[k] != S_['base'][k] for k in ref), 'of', len(ref))
