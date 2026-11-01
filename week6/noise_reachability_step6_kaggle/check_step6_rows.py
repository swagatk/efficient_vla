"""Step 6 row checks: merged files against parts, seeds 16-31 against seeds 0-15, per-state counts (argument: run folder)."""
import csv,glob,json,sys,collections
D=sys.argv[1]
rd=lambda f:list(csv.DictReader(open(f)))
base=rd(D+'/eval/per_episode_base.csv'); b=rd(D+'/eval/per_episode_B.csv')
key=lambda r:(int(r['task_id']),int(r['seed']),int(r['init_state']))
B={key(r):r for r in b}; Bs={key(r):r for r in base}
assert len(B)==640==len(Bs) and set(B)==set(Bs)
print('tasks',sorted({k[0] for k in B}),'seeds',min(k[1] for k in B),max(k[1] for k in B),'states',sorted({k[2] for k in B}))
# merged == parts
for a,m in (('base',Bs),('B',B)):
    parts={}
    for f in glob.glob(f'{D}/eval/{a}/task_*/part*/per_episode_base.csv'):
        for r in rd(f):
            assert key(r) not in parts; parts[key(r)]=r
    print(a,'parts',len(parts),'equal merged',all(parts[k]==m[k] for k in m))
# old seeds 0-15
old={}
for f in ['week6/noise_reachability_step0_kaggle/nas10_all_tasks/per_episode.csv','week6/noise_reachability_step1_kaggle/nas10_step1/per_episode.csv']:
    for r in rd(f):
        if r.get('policy','base') not in('base',): continue
        t=int(r['task_id'])
        if t in(4,6,7,8): old[key(r)]=r
print('old rows',len(old),'seeds',sorted({k[1] for k in old})[:3],sorted({k[1] for k in old})[-3:])
s=lambda r:int(r['success'])
# seed-to-seed: does a new seed copy an old one? agreement of outcomes & steps with seed-16 shift
for t in (4,6,7,8):
    agree=sum(s(Bs[(t,sd+16,i)])==s(old[(t,sd,i)]) for sd in range(16) for i in range(10))
    print('task',t,'base new',sum(s(Bs[(t,sd,i)]) for sd in range(16,32) for i in range(10)),'old',sum(s(old[(t,sd,i)]) for sd in range(16) for i in range(10)),'same outcome',agree,'/160')
    print('   B vs base: differ outcome',sum(s(B[(t,sd,i)])!=s(Bs[(t,sd,i)]) for sd in range(16,32) for i in range(10)),'differ steps',sum(B[(t,sd,i)]['steps']!=Bs[(t,sd,i)]['steps'] for sd in range(16,32) for i in range(10)))
print('step-table: state: old/16 base/16 B/16')
sysS={4:[2],6:[8],8:[4]}; low={4:[1,5,6,7,8],6:[0,1,9],8:[1,2,3,8,9]}
G=[];Sg=[]
for t in (4,6,7,8):
    print('task',t)
    for i in range(10):
        o=sum(s(old[(t,sd,i)]) for sd in range(16)); a=sum(s(Bs[(t,sd,i)]) for sd in range(16,32)); c=sum(s(B[(t,sd,i)]) for sd in range(16,32))
        tag='SYS' if i in sysS.get(t,[]) else 'low' if i in low.get(t,[]) else ''
        print(f'  {i}: {o:2d} {a:2d} {c:2d} {tag}')
        if tag=='SYS':Sg.append((t,(c-a)/16))
        if tag=='low':G.append((t,(c-a)/16))
print('mean low gain',sum(g for _,g in G)/len(G),len(G))
pt={t:sum(g for u,g in G if u==t)/len([1 for u,_ in G if u==t]) for t in (4,6,8)}
print('per task low',pt,'sys',Sg)
print('D (low mean-of-task minus sys)',sum(pt[t]-dict(Sg)[t] for t in pt)/3)
from math import comb
def sign_p(a,b):
    n=a+b; k=min(a,b)
    return min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n)
print('largest changes by state, paired by seed: task state base B gained lost p')
rows=[]
for t in (4,6,7,8):
    for i in range(10):
        up=sum(s(B[(t,sd,i)])>s(Bs[(t,sd,i)]) for sd in range(16,32)); dn=sum(s(B[(t,sd,i)])<s(Bs[(t,sd,i)]) for sd in range(16,32))
        rows.append((t,i,sum(s(Bs[(t,sd,i)]) for sd in range(16,32)),sum(s(B[(t,sd,i)]) for sd in range(16,32)),up,dn,sign_p(up,dn)))
for r in sorted(rows,key=lambda r:r[-1])[:8]: print(' ',r[0],r[1],r[2],r[3],r[4],r[5],round(r[6],5))
print('states of 40 with p<.05 before correction:',sum(r[-1]<.05 for r in rows),'; Bonferroni .05/40:',sum(r[-1]<.05/40 for r in rows))
print('totals, 4 tasks: base',sum(s(r) for r in base),'B',sum(s(r) for r in b),'of 640')
base_all={k:s(r) for k,r in old.items()}
print('state 4/2, base on 32 seeds:',sum(base_all[(4,sd,2)] for sd in range(16))+sum(s(Bs[(4,sd,2)]) for sd in range(16,32)),'/32')
