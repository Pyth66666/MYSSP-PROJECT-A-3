import json,statistics,collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=json.load(open('runs.json'))
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; SURF='#ffffff'
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':INK2,'axes.labelcolor':INK,
  'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':SURF,'axes.facecolor':SURF})
M=['gpt-6-sol','gpt-6-luna','gpt-5.6-terra','gpt-5.5','gpt-5.4','gpt-5.4-mini','gpt-5.4-nano','deepseek-v4-flash','deepseek-v4.1-flash']
# open-source models, design A only (design C not run yet): aggregates from the laptop run logs, 50 runs each
OSM={'qwen2.5:3b':([98,24,0,0,0],11177,32.9),'llama3.2:3b':([96,26,0,0,0],22808,53.1),'gemma2:2b':([100,50,0,0,0],30276,57.0),'phi3.5:3.8b':([4,2,0,0,0],26313,67.8)}
AC=[r for r in rows if r['design'] in 'AC']
def sub(m,d=None): return [r for r in AC if r['model']==m and (d is None or r['design']==d)]
pct=lambda s,i: 100*sum(r['m'][i] for r in s)/len(s)
# order models by pass rate (milestone 4)
order=sorted(M,key=lambda m:(pct(sub(m),3),pct(sub(m),4)))
osorder=sorted(OSM,key=lambda m:OSM[m][0][1])
order=osorder+order      # open-source models at the bottom, commercial above
def P(m,i): return OSM[m][0][i] if m in OSM else pct(sub(m),i)
def TK(m,d=None):
    if m in OSM: return OSM[m][1] if d in (None,'A') else None
    return statistics.median([r['tokens'] for r in sub(m,d)])
def SC(m): return OSM[m][2] if m in OSM else statistics.median([r['seconds'] for r in sub(m)])
# ---- Figure 1: token usage per run, design A vs C
fig,ax=plt.subplots(figsize=(6.4,3.6),dpi=200)
y=range(len(order)); h=0.36
for j,(d,c,lab) in enumerate([('A',BLUE,'Design A (no noise)'),('C',ORANGE,'Design C (3% noise)')]):
    v=[(TK(m,d) or 0)/1000 for m in order]
    ax.barh([i+(h/2+0.02)*(1 if j==0 else -1) for i in y],v,height=h,color=c,label=lab,zorder=2)
    for i,val in zip(y,v):
        if val>0: ax.text(val+0.2,i+(h/2+0.02)*(1 if j==0 else -1),f'{val:.1f}',va='center',fontsize=7,color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels(order)
ax.set_xlabel('Median tokens per run (thousands)'); ax.set_xlim(0,34)
ax.grid(axis='x',color=GRID,lw=0.6,zorder=0); ax.tick_params(axis='y',length=0)
ax.legend(frameon=False,loc='upper right',fontsize=8)
ax.axhline(3.5,color=GRID,lw=0.8)
ax.set_title('Token use per run (bottom four: open-source, design A only)',loc='left',fontsize=10,color=INK)
fig.tight_layout(); fig.savefig('fig_tokens.png'); fig.savefig('fig_tokens.pdf')
# ---- Figure 2: milestones and resources, rows aligned
fig,axs=plt.subplots(1,3,figsize=(8.6,3.6),dpi=200,gridspec_kw={'width_ratios':[1.6,1,1]},sharey=True)
ax=axs[0]
for i_m,(i,c,mk,lab) in enumerate([(0,'#9a9893','o','M1 Answer: formula readable'),(1,'#5f5e5a','D','M2 Shape: error below 100%'),(2,BLUE,'o','M3 Close: error 10% or less'),(3,ORANGE,'s','M4 Pass: within design tolerance'),(4,AQUA,'^','M5 Exact: SymPy match')]):
    v=[P(m,i) for m in order]
    ax.scatter(v,list(y),s=34,color=c,marker=mk,label=lab,zorder=3,edgecolors=SURF,linewidths=1.2)
for k,m in enumerate(order):
    vs=[P(m,i) for i in range(5)]
    ax.plot([min(vs),max(vs)],[k,k],color=GRID,lw=2,zorder=1)
ax.set_xlim(0,105); ax.set_xlabel('Runs reaching milestone (%)')
ax.set_yticks(list(y)); ax.set_yticklabels(order); ax.tick_params(axis='y',length=0)
ax.grid(axis='x',color=GRID,lw=0.6,zorder=0)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.2),fontsize=7,ncol=2)
ax.set_title('Milestones reached',loc='left',fontsize=10,color=INK)
for ax,key,lab,title,fmt in [(axs[1],'tokens','Median tokens per run (thousands)','Tokens',lambda v:v/1000),(axs[2],'seconds','Median seconds per run','Time',lambda v:v)]:
    v=[fmt(TK(m)) if key=='tokens' else SC(m) for m in order]
    ax.barh(list(y),v,height=0.55,color=BLUE,zorder=2)
    for i,val in zip(y,v): ax.text(val*1.02,i,f'{val:.1f}',va='center',fontsize=7,color=INK2)
    ax.set_xlabel(lab); ax.grid(axis='x',color=GRID,lw=0.6,zorder=0); ax.tick_params(axis='y',length=0)
    ax.set_title(title,loc='left',fontsize=10,color=INK); ax.set_xlim(0,max(v)*1.25)
fig.text(0.01,0.005,'Open-source models (bottom four): design A only, 50 runs each. Other models: designs A and C, 100 runs each.',fontsize=7,color=INK2)
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig('fig_milestones_tokens.png'); fig.savefig('fig_milestones_tokens.pdf')
print('ok')
