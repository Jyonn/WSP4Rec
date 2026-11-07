"""Scientific figures from the completed, frozen v7 text study."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/v7_text'
rows=list(csv.DictReader((R/'per_case.csv').open()));summary=json.loads((R/'summary.json').read_text());seeds=(42,43,44)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def vals(access,method):return [np.mean([float(r['ndcg@10']) for r in rows if r['access']==access and r['method']==method and int(r['seed'])==seed]) for seed in seeds]
fig,axes=plt.subplots(1,3,figsize=(13,4.1),sharey=True)
# Main three-way comparison, same features for each checkpoint's methods.
colors=('#a5aaa4','#bd995d','#257b5c');methods=('bp','fixed_simple','astra');names=('BP','Fixed SVD+BP','Astra')
for x,access in enumerate(('none','full_text')):
 for j,(method,color) in enumerate(zip(methods,colors)):
  v=vals(access,method);position=x+(j-1)*.24
  axes[0].bar(position,np.mean(v),width=.22,color=color,label=names[j] if x==0 else None)
  axes[0].scatter(position+np.linspace(-.04,.04,3),v,color='#29352e',s=13,zorder=3)
axes[0].set_xticks((0,1),('Without text','Text + readable descriptions'));axes[0].tick_params(axis='x',labelsize=9);axes[0].set_title('Main comparison');axes[0].legend(frameon=False,fontsize=8,loc='upper left');axes[0].set_ylabel('Test NDCG@10')
# Proper text-to-item correspondence control.
bp_none=vals('none','bp');bp_true=vals('full_text','bp');bp_shuffle=[summary['bp_only_shuffled_text']['ndcg@10']]
final=json.loads((ROOT/'runs/v7_text/final_results.json').read_text());bp_shuffle=[final['bp_only_shuffled_text'][str(s)]['summary']['ndcg@10'] for s in seeds]
for x,v,color in zip(range(3),(bp_none,bp_true,bp_shuffle),('#a5aaa4','#257b5c','#bd995d')):
 axes[1].bar(x,np.mean(v),color=color,width=.65);axes[1].scatter(x+np.linspace(-.1,.1,3),v,color='#29352e',s=14,zorder=3)
axes[1].set_xticks(range(3),('No text','Correct text','Shuffled text'),rotation=20,ha='right');axes[1].set_title('BP feature correspondence')
# Raw-readable text and numeric features use exactly the same text BP states.
for index,access in enumerate(('numeric_text','full_text')):
 v=vals(access,'astra');axes[2].bar(index,np.mean(v),color=('#879eab','#257b5c')[index],width=.6)
 for seed,color in zip(seeds,('#9a6338','#536baf','#477a50')):
  values=[float(r['ndcg@10']) for r in rows if r['access']==access and r['method']=='astra' and int(r['seed'])==seed]
  axes[2].scatter(index+np.linspace(-.10,.10,2),values,s=18,color=color,zorder=3)
for seed,color in zip(seeds,('#9a6338','#536baf','#477a50')):
 axes[2].plot((0,1),(vals('numeric_text','astra')[seeds.index(seed)],vals('full_text','astra')[seeds.index(seed)]),color=color,alpha=.4,linewidth=1)
axes[2].set_xticks((0,1),('Numeric text vectors','Also readable text'),rotation=15,ha='right');axes[2].set_title('Astra text access')
for ax in axes:ax.set_ylim(0,.16);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Frozen Llama-3 title + description features: matched BP and editing comparisons',fontsize=12)
fig.tight_layout(rect=(0,.02,1,.92));fig.savefig(R/'text_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(R/'text_comparison.pdf',bbox_inches='tight')
