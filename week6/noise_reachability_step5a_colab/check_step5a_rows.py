import csv,glob,json,os,re,sys,random,math,collections as C
random_seed_for_permutation = 0
R=sys.argv[1]; random.seed(random_seed_for_permutation)
def rows(p): return list(csv.DictReader(open(p)))
fam={}
for p in sorted(glob.glob(R+'/per_episode_*_state*.csv')):
    m=re.search(r'per_episode_(\w+)_state(\d)\.csv',p); f,s=m[1],int(m[2]); rr=rows(p)
    fam[f,s]=rr
    seeds=[int(r['seed']) for r in rr]
    n=35 if s==9 else 70
    assert sorted(seeds)==[1000+i for i in range(n)],(p,len(seeds))
    assert all(int(r['init_state'])==s and r['task_id']=='0' and r['success'] in '01' for r in rr),p
print('files',len(fam),'episodes',sum(len(v) for v in fam.values()))
print('state  sampled full repeat')
tot=C.Counter()
hits=C.defaultdict(list)
for s in (2,5,6,7,8,9):
    o=[]
    for f in ('sampled','full','repeat'):
        h=[(int(r['seed'])-1000,int(r['steps'])) for r in fam[f,s] if r['success']=='1']
        o.append(len(h)); hits[f,s]=h
        if s!=9: tot[f]+=len(h)
    print(s,o,len(fam['full',s]))
print('systematic totals of 350',dict(tot))
print('fail-step check',C.Counter(int(r['steps']) for k,v in fam.items() for r in v if r['success']=='0'))
for k,v in sorted(hits.items()):
    if k[1]!=9 and v: print(k,v)
# noise ids across states
for f in ('full','repeat'):
    c=C.Counter(i for s in (2,5,6,7,8) for i,_ in hits[f,s])
    obs=sum(1 for v in c.values() if v>=2)
    sim=[]
    for _ in range(20000):
        d=C.Counter()
        for s in (2,5,6,7,8):
            for i in random.sample(range(70),len(hits[f,s])): d[i]+=1
        sim.append(sum(1 for v in d.values() if v>=2))
    print(f,'ids hitting >=2 states',obs,{k:v for k,v in c.items() if v>=2},'perm p',sum(x>=obs for x in sim)/len(sim),'mean',sum(sim)/len(sim))
# same id across families (full vs repeat use different noise?)
# fisher
def fisher(a,n,b,m):
    from math import comb
    t=a+b; N=n+m
    p0=comb(n,a)*comb(m,b)/comb(N,t)
    return sum(comb(n,i)*comb(m,t-i)/comb(N,t) for i in range(0,min(t,n)+1) if t-i<=m and comb(n,i)*comb(m,t-i)/comb(N,t)<=p0*(1+1e-9))
for f in ('full','repeat'): print('fisher',f,'vs sampled',tot[f],0,fisher(tot[f],350,0,350))
print('fisher repeat vs full',fisher(tot['repeat'],350,tot['full'],350))
def cp(k,n,a=.05):
    def bin(k,n,p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
    lo=0;hi=1
    if k:
        l,h=0,1
        for _ in range(60):
            m=(l+h)/2
            if 1-bin(k-1,n,m)<a/2:l=m
            else:h=m
        lo=l
    if k<n:
        l,h=0,1
        for _ in range(60):
            m=(l+h)/2
            if bin(k,n,m)>a/2:l=m
            else:h=m
        hi=h
    return round(lo,4),round(hi,4)
for f in tot: print('CI',f,tot[f],cp(tot[f],350))
print('CI ctrl',{f:(sum(r['success']=='1' for r in fam[f,9]),cp(sum(r['success']=='1' for r in fam[f,9]),35)) for f in tot})
# confirmations
tj=C.Counter();
print('confirm:')
for p in sorted(glob.glob(R+'/confirm_*.csv')):
    m=re.search(r'confirm_(\w+)_n(\d+)_state(\d)',p); rr=rows(p)
    assert sorted(int(r['seed']) for r in rr)==[5000,5001,5002,5003],p
    k=sum(r['success']=='1' for r in rr); print(m[1],m[2],'state',m[3],k,'/4',[r['steps'] for r in rr if r['success']=='1'])
    tj[m[1],'k']+=k;tj[m[1],'n']+=4
print(tj)
# provenance
pr=[]
for p in [R+'/provenance.jsonl']+sorted(glob.glob(R+'/harness/seed*/provenance.jsonl')):
    pr+= [json.loads(l) for l in open(p)]
print('prov records',len(pr),'keys',sorted(pr[0])[:30])
for k in ('git_commit','git_dirty','git_dirty_noise_reachability','policy_path'):
    print(k,C.Counter(str(x.get(k, x.get('args',{}).get(k))) for x in pr))
print(open(R+'/repo_commit.txt').read().strip())
a=pr[0].get('args') or pr[0].get('config') or {}
print(a)
for k in ('n_action_steps','max_steps','rename_map','task_id','policy_path'):
    print(k,C.Counter(str((x.get('args') or x.get('config') or {}).get(k)) for x in pr))
