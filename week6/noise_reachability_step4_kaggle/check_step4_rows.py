import csv,glob,json,sys,os,collections,hashlib
D=sys.argv[1]
def rows(p): return list(csv.DictReader(open(p)))
key=lambda r:(int(r['seed']),int(r['init_state']))
print('columns',rows(D+'/eval/per_episode_base.csv')[0].keys())
arms='base A0 A1 A2 B C0 C1 C2'.split()
R={}
for a in arms:
    m={key(r):r for r in rows(f'{D}/eval/per_episode_{a}.csv')}
    parts={}
    for p in glob.glob(f'{D}/eval/{a}/part*/per_episode_*.csv'):
        for r in rows(p):
            assert key(r) not in parts; parts[key(r)]=r
    assert {k:v['success'] for k,v in m.items()}=={k:v['success'] for k,v in parts.items()},a
    R[a]=m
print('merged == parts for all arms')
S={2,5,6,7,8}
for a in arms:
    d=sum(R[a][k]['success']!=R['base'][k]['success'] for k in R['base'])
    sy=[k for k in R[a] if k[1] in S and int(R[a][k]['success'])]
    st=[k for k in R['base'] if 'steps' in R[a][k] and R[a][k]['steps']!=R['base'][k]['steps']]
    print(a,'succ',sum(int(r['success']) for r in R[a].values()),'differs outcome',d,'differs steps',len(st),'systematic hits',sorted(sy))
# provenance
c=collections.Counter()
for p in glob.glob(D+'/eval/*/part*/provenance.jsonl')+glob.glob(D+'/harness_check/part*/provenance.jsonl'):
    a=p.split('/')[-3]
    for l in open(p):
        r=json.loads(l)
        c[(a,r.get('git_commit','')[:7],r.get('git_dirty'),r.get('git_dirty_noise_reachability'),str(r.get('gpu') or r.get('cuda_device') or ''))]+=1
for k,v in sorted(c.items()): print(k,v)
r=json.loads(open(glob.glob(D+'/eval/C0/part0/provenance.jsonl')[0]).readline()); print(sorted(r.keys())); print({k:v for k,v in r.items() if k in('argv','config','command','args')})
r=json.loads(open(glob.glob(D+'/eval/B/part0/provenance.jsonl')[0]).readline()); print({k:v for k,v in r.items() if k in('argv','config','command','args')})
for f in sorted(glob.glob(D+'/**/adapter_model*',recursive=True)): print(f.replace(D,''))
for f in sorted(glob.glob(D+'/**/run_config.json',recursive=True)):
    j=json.load(open(f)); print(f.replace(D,''),{k:j.get(k) for k in j if k in('seed','steps','max_steps','data_dir','lr','episodes','n_demos')})
for f in sorted(glob.glob(D+'/**/train_history.json',recursive=True)):
    h=json.load(open(f)); h=h if isinstance(h,list) else h.get('history',h)
    print(f.replace(D,''),len(h),str(h[0])[:110],str(h[-1])[:110])
print(open(D+'/data/collection.csv').readline().strip()); cr=rows(D+'/data/collection.csv'); print(len(cr),sum(int(r['success']) for r in cr))
print(json.load(open(D+'/demos/demo_manifest.json')).keys() if isinstance(json.load(open(D+'/demos/demo_manifest.json')),dict) else 'list')
