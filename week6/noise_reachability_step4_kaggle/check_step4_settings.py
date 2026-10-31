import json,glob,sys,csv,collections
D=sys.argv[1]
for a in 'base A0 A1 A2 B C0 C1 C2'.split():
    vals=collections.defaultdict(set)
    for p in glob.glob(f'{D}/eval/{a}/part*/provenance.jsonl'):
        for l in open(p):
            r=json.loads(l)
            for k,v in r['settings'].items():
                if k not in('seeds','out_dir','init_states','output_dir'): vals[k].add(json.dumps(v)[:70])
            vals['_dirty_sfbc'].add(r.get('git_dirty_sfbc')); vals['_gpus'].add(str(r.get('gpus'))[:50]); vals['_date'].add(r['date'][:10])
    print(a,{k:sorted(map(str,v)) for k,v in vals.items() if len(v)>1 or k in('n_action_steps','policy_path','lora_dir','adapter','adapter_dir','task_id','_dirty_sfbc','_date')})
    if a=='C0': print('  keys',sorted(vals))
for a in 'A0 A1 A2 C0 C1 C2'.split():
    h=json.load(open(f'{D}/runs/{a}/train_history.json')); h=h if isinstance(h,list) else h['history']
    l=[x['loss'] for x in h]; n=8
    print(a,'loss first8 %.4f last8 %.4f'%(sum(l[:n])/n,sum(l[-n:])/n))
    c=json.load(open(f'{D}/runs/{a}/adapter/run_config.json')); print('  ',{k:c[k] for k in list(c)[:14]} )
m=json.load(open(D+'/demos/demo_manifest.json')); print({k:m[k] for k in('task','task_index','total_frames','frame_stride')},len(m['episodes']))
# per-state table
S=[2,5,6,7,8]
st=collections.defaultdict(dict)
for a in 'base A0 A1 A2 B C0 C1 C2'.split():
    for r in csv.DictReader(open(f'{D}/eval/per_episode_{a}.csv')):
        st[a][int(r['init_state'])]=st[a].get(int(r['init_state']),0)+int(r['success'])
print('state  '+' '.join(f'{i:>3}' for i in range(10)))
for a in st: print(f'{a:6} '+' '.join(f'{st[a].get(i,0):>3}' for i in range(10)))
