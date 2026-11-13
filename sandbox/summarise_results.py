"""Per (law, design, model) summary from results/experiments.tsv.
Same rule as the 3D leaderboard: 10 latest runs, RSI runs left out, runs that
failed only because the internet or API connection dropped left out."""
import csv,statistics,collections
NET=("urlopen error","getaddrinfo","timed out","Connection","RemoteDisconnected")
rows=list(csv.DictReader(open('results/experiments.tsv',encoding='utf-8'),delimiter='\t'))
g=collections.defaultdict(list)
for r in rows:
    if r['label']!='experiment' or not r['agent'].startswith('RapidScreen'): continue
    a=r['answer'] or ''
    if a.startswith('agent error') and any(e in a for e in NET): continue
    g[(r['law'],r['design'],r['agent'][12:])].append(r)
out=[]
for k,v in sorted(g.items()):
    v=sorted(v,key=lambda r:r['timestamp'])[-10:]
    t=lambda x:x=='True'
    errs=[float(r['nrmse']) for r in v if r['nrmse'] not in ('','None')]
    top=collections.Counter((r['answer'] or '')[:70] for r in v).most_common(1)[0]
    out.append(dict(law=k[0],design=k[1],model=k[2],runs=len(v),passed=sum(t(r['passed']) for r in v),
      exact=sum(t(r['exact']) for r in v),median_err=round(statistics.median(errs)*100,2) if errs else '',
      blocked=sum('HTTP 400' in (r['answer'] or '') for r in v),top_answer=top[0],top_count=top[1],
      avg_exp=round(sum(int(r['experiments_used'] or 0) for r in v)/len(v),1),
      tokens=sum(int(float(r['tokens'] or 0)) for r in v),first=v[0]['timestamp'],last=v[-1]['timestamp']))
w=csv.DictWriter(open('results/summary.tsv','w',newline='',encoding='utf-8'),fieldnames=list(out[0]),delimiter='\t')
w.writeheader(); w.writerows(out)
for r in out:
    if r['law'] in ('power_spring','kinetic_energy','wave_speed','screened_coulomb','saturating_ohm'):
        print(r['law'][:10],r['design'],f"{r['model']:<20}",r['runs'],r['passed'],r['exact'],r['median_err'],r['blocked'],'|',r['top_count'],r['top_answer'][:40])
