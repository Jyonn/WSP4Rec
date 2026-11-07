"""Build a standalone Chinese research report from frozen results only."""
import json,html,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'runs/v8_mit';R=ROOT/'reports/v8_mit'
def read(p):return json.loads(p.read_text())
f=read(OUT/'final_results.json');br=read(OUT/'real_bridge/final_results.json');cert=read(OUT/'formal_certificates.json');cost=read(R/'costs.json');ci=read(R/'paired_user_intervals.json');cases=f['cases'];base=f['baselines']
def mean(xs):return float(np.mean(xs))
def bm(task,n,condition,s):return mean([base[f'task{task}-n{n}-s{seed}-{condition}']['metrics'][s]['ndcg@10'] for seed in (42,43,44)])
def rm(budget,s):return mean([v['metrics'][s]['ndcg@10'] for v in cases.values() if v['assignment']['kind']=='repair' and v['assignment']['extra_forward_budget']==budget])
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.facecolor':'#faf8f2','figure.facecolor':'#faf8f2','axes.prop_cycle':plt.cycler(color=['#a43c2d','#235955','#bf9145','#67605a'])})
fig,axes=plt.subplots(1,3,figsize=(11,3.3),sharey=True)
for t,ax in enumerate(axes):
 for s,label in [('warm_warm','Warm held-out pairs'),('both_cold','Both entities cold')]:ax.plot([128,2048,16384],[bm(t,n,'warm_only',s) for n in (128,2048,16384)],'o-',label=label)
 ax.axhline(1,color='#67605a',ls='--',lw=1,label='Astra / exact compiler');ax.set_xscale('log');ax.set_title(f'Policy {t+1}');ax.set_xlabel('BP training pairs');ax.set_ylim(0,1.04);ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('Graded policy NDCG@10');axes[0].legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(R/'data_efficiency.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.4));ks=[0,5,10]
for name,label in [('bp','BP content ranker (3 seeds)'),('astra','Astra = fixed compiler (4 programs)'),('genre_profile_cosine','Genre profile cosine')]:
 vals=[]
 for k in ks:
  m=br['budgets'][str(k)]['metrics'];vals.append(mean([v['ndcg@10'] for n,v in m.items() if n.startswith('bp-')]) if name=='bp' else m['fixed_symbolic_task0' if name=='astra' else name]['ndcg@10'])
 ax.plot(ks,vals,'o-',label=label)
ax.set_xticks(ks);ax.set_xlabel('Onboarding warm-item ratings supplied (K)');ax.set_ylabel('Binary real-interaction NDCG@10');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(R/'real_bridge.svg');plt.close(fig)
slices=['warm_warm','cold_user','cold_item','both_cold'];repairrows=[]
for cond,label in [('warm_only','普通 BP'),('warm_only-fixed_fold','固定状态修复'),('cold_aware','冷启动增强 BP')]:repairrows.append([label]+[f'{bm(1,16384,cond,s):.5f}' for s in slices])
repairrows +=[[f'Astra：已有行为样例，额外 {budget} 批反馈']+[f'{rm(budget,s):.5f}' for s in slices] for budget in (0,2)];repairrows.append(['固定规则编译器']+['1.00000']*4)
realrows=[]
for label,selector in [('BP（三种子均值）','bp'),('Astra（四个冻结网络）','fixed_symbolic_task0'),('固定规则编译器','fixed_symbolic_task0'),('连续题材偏好余弦','genre_profile_cosine'),('训练集总体题材先验','global_genre_positive_rate'),('常数分数／固定 ID 解平局','constant_score_reference')]:
 vals=[]
 for k in ks:
  m=br['budgets'][str(k)]['metrics'];v=mean([v['ndcg@10'] for n,v in m.items() if n.startswith('bp-')]) if selector=='bp' else m[selector]['ndcg@10'];vals.append(f'{v:.5f}')
 realrows.append([label]+vals)
writerows=[]
for t in range(3):
 for enc,label in [('natural','自然名称'),('nonce','新符号＋完整含义表＋重排输入')]:
  es=[e for e in cases.values() if e['assignment']['kind']=='write' and e['assignment']['task']==t and e['assignment']['encoding']==enc]
  writerows.append([t+1,label,'2 / 2',f"{mean([e['metrics']['both_cold']['ndcg@10'] for e in es]):.5f}",'2 / 2'])
allrows=[]
for key,v in cases.items():
 a=v['assignment'];m=v['metrics']['both_cold'];allrows.append([key,'从零写入' if a['kind']=='write' else '已有模型重写',a.get('encoding',f"seed{a['seed']}，额外{a['extra_forward_budget']}批") if 'seed' in a else a['encoding'],f"{m['ndcg@10']:.5f}",f"{m['utility_error']['max_absolute']:.2e}",v.get('changed_parameters','—')])
abl=[]
for mode,label in [('user_missing','删除用户偏好'),('item_missing','删除物品属性'),('both_missing','两者均删除，仅剩无关状态'),('misaligned','输入含义错位')]:abl.append([label,f"{mean([e['information_ablations'][mode]['ndcg@10'] for e in cases.values() if 'information_ablations' in e]):.5f}"])
models={}
for key in ('w05','r01'):
 a=cases[key]['assignment']
 with np.load(ROOT/a['path']) as z:models[key]={k:z[k].tolist() for k in z.files}
 if key=='r01':
  with np.load(ROOT/a['source']) as z:models['BP']={k:z[k].tolist() for k in z.files}
answer=(OUT/'cases/w05/turn0/answer.txt').read_text();edit=answer.split('```edits')[1].split('```')[0].strip()
maxerror=max(e['metrics'][s]['utility_error']['max_absolute'] for e in cases.values() for s in slices)
research_tokens=cost['research_token_counters'];interval=ci['real-K10-astra-mean-minus-bp-mean']
source='https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WSP4Rec · 把规则写进权重之后</title><style>
:root{--paper:#faf8f2;--ink:#282521;--muted:#685e53;--red:#a43c2d;--line:#d8d0c1;--green:#235955}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:17px/1.9 'PingFang SC','Microsoft YaHei',sans-serif}main{max-width:990px;margin:auto;padding:44px 28px 90px}header{border-top:5px solid var(--red);padding-top:24px;margin-bottom:50px}.eyebrow{font:13px/1.6 Georgia,serif;letter-spacing:.1em;color:var(--red)}h1{font:600 clamp(32px,5.3vw,58px)/1.22 'Songti SC',serif;margin:23px 0}h2{font:600 30px/1.4 'Songti SC',serif;margin:62px 0 18px;border-top:1px solid var(--line);padding-top:24px}h3{font-size:20px;margin-top:34px}.lead{font-size:21px;max-width:830px}a{color:var(--red);text-underline-offset:4px}p{margin:15px 0}strong{font-weight:650}small,.caption{font-size:14px;color:var(--muted)}.verdict{border-left:4px solid var(--green);padding:5px 22px;margin:28px 0;background:#efeee4}.scroll{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line)}th{background:#ece7dc}figure{margin:28px 0}figure img{width:100%;height:auto}pre{padding:20px;background:#eee9dd;overflow:auto;font:13px/1.65 Menlo,monospace;border-left:3px solid var(--red)}code{font-size:.85em}details{border-bottom:1px solid var(--line);padding:14px 0}summary{cursor:pointer;font-weight:600}.controls{display:grid;grid-template-columns:1fr 1fr;gap:18px}.controls fieldset{border:1px solid var(--line);min-width:0;padding:15px}.controls label{display:block;font-size:14px;line-height:2.1}.controls select{font:inherit}input[type=range]{max-width:100%}.outputs{display:flex;gap:25px;flex-wrap:wrap;padding:18px 0;border-bottom:1px solid var(--line)}.outputs b{font:600 25px Georgia,serif;display:block}.network{width:100%;height:auto}.links{font-size:14px}footer{margin-top:50px;border-top:1px solid var(--line);padding-top:15px;color:var(--muted);font-size:13px}@media(max-width:600px){main{padding:25px 18px 60px}body{font-size:16px}.controls{grid-template-columns:1fr}h2{font-size:26px}th,td{padding:9px}header{margin-bottom:30px}}
</style><main><header><div class="eyebrow">WSP4REC / EXPERIMENT NOTE 08 / 2026.11.08</div><h1>把规则写进权重之后，<br>它懂了用户吗？</h1><p class="lead">Astra 确实能不训练、直接写出正确的推荐网络。这个能力可以被逐项检验，甚至被数学证明；但在真实电影偏好预测上，现有结果仍然支持 BP。</p><p class="caption">36 次原生 Astra 请求 · 24 个最终网络 · 33 次 BP 训练 · A100 服务器 · 先封存、后测试</p></header>
<div class="verdict"><strong>本轮的核心结论</strong><p>已知规则能被写入网络，未知偏好不会因此凭空出现。十二次“修复”都重新写了整个网络，所以收益应归因于规则重编译，不能称为保留原模型知识的局部修复。</p></div>
<h2>01 / MIT 到底给了 Astra 什么？</h2><p>官方仓库的共同做法是：说明输入、输出和网络结构的含义，提供任务规则；修复已有模型时再提供权重和失败行为。它没有证明 Astra 只看一堆匿名数字就能理解任意模型。</p>
'''
page+=table(['MIT 示例','Astra 可见信息与预算','需要保留的边界'],[
['从零写游戏网络','明确游戏源码与语义输入；一次回复、无工具、零前向','Breakout 很强；另两款游戏没有全面胜过更强 PPO'],
['修复小车','权重、固定场景、一次失败轨迹；一次文字编辑','同一场景的随机动作测试，不是新场景泛化'],
['围棋','棋盘语义与规则；0 / 10 / 25 / 50 批反馈','零反馈没有击败调好的同尺寸 PPO'],
['修复 Qwen','练习题、输出与激活；受控前向反馈','主要救回循环截断；Gemma 后续实验未证实可靠收益']])
page+=f'''<p class="links">逐项依据：<a href="{source}exp5_write_from_scratch/README.md">游戏</a> · <a href="{source}exp4_fix_car/README.md">小车</a> · <a href="{source}exp6_go_from_scratch/README.md">围棋</a> · <a href="{source}exp8_math/README.md">Qwen</a> · <a href="{source}exp8_math/gemma_followup/README.md">Gemma 负结果</a>。本轮锁定官方提交 4d09ba6；不是复用它发布的权重改动。</p>
<h2>02 / 全新用户、全新物品，需要什么前提？</h2><p>假设电影 X 和用户 A 从未出现在 LLM 的训练中。如果知道 A 喜欢科幻、不能接受闪烁画面，也知道 X 是科幻片且含闪烁画面，仍然可以依据规则排序。如果只知道两个随机 ID，就无法辨认 A 的私人喜好。</p><p>因此，本轮生成全新的虚构用户和物品，提供可解释的偏好与内容属性。身份与组合在协议锁定后生成；最终冷启动实体从未给 Astra 看过。通用题材概念、算术、神经网络构造可能来自预训练，实验没有排除这些知识，也没有声称能够核查专有预训练语料。</p><p><strong>这里的冷启动是“目标实体没有训练交互，但有偏好／内容资料”。新身份不等于未知偏好。</strong>真实电影的预训练成员身份未知；屏蔽片名与 ID 只能阻止界面上的实体检索。</p>
<h2>03 / 本轮可调整的网络</h2><p>模型有 1,825 个可编辑数字。它接收 22 个有明确含义的特征，经过两层各 32 个神经元，输出一个推荐分数。这轮没有用户／电影 ID embedding 表，也没有重新使用 Llama 简介向量；这些属于此前 v7 的另一套实验。</p>
<svg class="network" viewBox="0 0 900 230" role="img" aria-label="用户属性和物品属性拼接为22维，进入两层32神经元网络，输出推荐分数"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#a43c2d"/></marker></defs><g fill="none" stroke="#b5aa97" stroke-width="1.5"><rect x="5" y="25" width="180" height="175"/><rect x="225" y="65" width="120" height="95"/><rect x="390" y="50" width="140" height="125"/><rect x="575" y="50" width="140" height="125"/><rect x="760" y="65" width="130" height="95"/><path d="M185,112 H222 M345,112 H387 M530,112 H572 M715,112 H757" stroke="#a43c2d" marker-end="url(#arrow)"/></g><g fill="#282521" text-anchor="middle" font-size="17"><text x="95" y="62">用户偏好 × 10</text><text x="95" y="102">物品属性 × 10</text><text x="95" y="142">是否训练中出现 × 2</text><text x="95" y="180" font-size="13">身份 ID 本身不输入</text><text x="285" y="105">拼接为 22 维</text><text x="285" y="138" font-size="13">固定含义表</text><text x="460" y="105">32 个 ReLU</text><text x="460" y="140" font-size="13">W1、b1</text><text x="645" y="105">32 个 ReLU</text><text x="645" y="140" font-size="13">W2、b2</text><text x="825" y="105">推荐分数</text><text x="825" y="138" font-size="13">W3、b3</text></g></svg>
<p>三个任务逐渐加入题材匹配、时间超限扣分、内容禁忌、纪录片例外和多条件奖励。规则及分值完整写在指令中。与 MIT 依据游戏规则发现策略相比，<strong>这里更接近把明确政策翻译成神经网络</strong>，难度与任务范围不能直接等同。</p>
<h2>04 / 一次回复，零次试跑：12 / 12 正确</h2><p>十二个全新 Astra 上下文，分别对应三个任务、两种输入编码、两次独立重复。自然名称之外，还把输入改成新符号、打乱位置，但给完整含义表。所有参数从零开始，只允许一次直接写数字的回复；没有训练数据、例子、工具、梯度、拟合或前向测试。</p>
'''
page+=table(['任务','输入形式','格式有效','全冷启动 NDCG@10','全域政策证书'],writerows)
page+=f'''<p>所有网络在四种切片均达到 NDCG@10 = 1，与固定规则编译器相同。独立求解器进一步检查所有 20 个二元输入以及连续的时长／时间预算范围：十二个网络全部未发现大于 10⁻⁶ 的实数算术误差，得到全域证明。数学证明把存储的 float32 参数当作精确实数；实际 float32 执行单独测试，最大效用误差为 {maxerror:.2e}，不能混称为浮点计算的逐位证明。</p><details><summary>查看 w05 的实际 Astra 权重回复</summary><p class="caption">这是自然名称、任务一的首个编号样例；没有按测试表现挑选。下面是模型直接返回的数字。</p><pre>{html.escape(edit)}</pre></details>
<h3>这些数字怎样执行一条规则？</h3><p>例如两个输入分别表示“用户喜欢科幻”和“物品是科幻片”。把相应两个连接设成 1，偏置设成 −1，就得到 ReLU(喜欢科幻＋是科幻−1)：只有两者都成立时才输出 1，再乘题材奖励。这是能直接核查的电路，不需要从交互中重新估计。</p>
<h3>把偏好或内容拿走，结果如何？</h3>'''
page+=table(['对冻结网络做的信息消融','全冷启动平均 NDCG@10'],abl)
page+='''<p class="caption">这是十二个冻结网络上的测试时信息删除／错位，不是另外训练的竞争模型。两侧信息删除后在本任务退化为同分排序，按固定 item ID 解平局，因此其结果相同。不能把这个数称作随机推荐期望，也不能据此证明匿名 ID 的所有推荐任务都不可能。</p><figure><img src="data_efficiency.svg" alt="三个政策下 BP 随训练数据量变化的暖实体和全冷启动排序结果"><figcaption class="caption">BP 每种设置三个种子，128 / 2,048 / 16,384 个训练对。相同架构、同一暖验证集选择检查点；Astra 和编译器已知精确规则，BP 从标签学习，二者获得信息的形式不同。曲线不是计算或信息完全等价的算法竞赛。</figcaption></figure>
<h2>05 / 修复已有 BP：分数变好，却不是局部修复</h2><p>选择任务二预先指定的三个高数据 BP 检查点，不按失败程度换模型。Astra 能看到全部权重、规则和一批 64 对行为样例（32 对暖验证＋32 对独立练习实体）；一半实验没有额外反馈，另一半最多再看两批各 128 对的候选前向结果（每批 64 对暖验证＋64 对独立练习，且不与前批样例重复）。没有告诉它固定修复公式，也没有告诉它故障类别。实际反馈是暖验证与练习的混合；原协议简写成“practice pairs”，这里按实现披露偏差，不含最终冷实体。</p><p>压力设置是人为设计的：普通 BP 训练时“已出现过”两列一直为 1，冷启动时变为 0。它检验模型对无关状态的敏感性，不能说真实推荐系统普遍有这一故障。经典修复只需把两列的贡献加到第一层偏置，再把两列置零；还训练了状态随机丢弃的冷启动增强 BP。</p>
'''
page+=table(['方法（三个源种子的均值）','暖用户／暖物品','冷用户','冷物品','双方都冷'],repairrows)
page+='''<p><strong>十二次 Astra 都改变了全部 1,825 个参数，写成可验证的政策网络。</strong>它们不是保留已学表示的稀疏补丁，两个额外反馈批次也没有产生更高的最终分数。十二个重写网络同样通过全域实数政策证明，但没有一个符合固定状态修复的结构。原 BP 暖实体排序接近正确，重写后暖排序为 1；数值预测没有保持不变，完整变化记录保存在逐次结果中。</p><p>固定状态修复已经把双方冷启动均值从 0.77219 恢复到 0.99429，且暖实体函数基本保留；冷启动增强 BP 达到 0.99403。Astra 达到 1 的主要解释是直接替换成精确已知规则。不能用这个差距主张它发现了 BP 的隐藏知识，也不能用大范围重写下的块回滚消融宣称定位了局部机制。</p>
<h2>06 / 真电影、真交互：写对规则还不够</h2><p>将任务一四个冻结网络直接应用到既有 1,568 部电影的数据子集。用全新的用户／物品划分，留下 314 部冷物品；它们不进入训练或用户支持历史。提供四类题材，其他未知内容字段使用中性值，不臆测电影的暴力、闪烁或时长。</p><p>同一批 189 名冷用户均有至少 10 次暖物品支持记录；评分用其中 167 名在共同时间截点后有正目标的用户。K=0 隐藏全部支持记录，K=5/10 提供最早 5/10 次；目标和候选集完全相同。K&gt;0 是少样本接入，K=0 是对这一已有历史群体隐藏证据，不是现实中从未交互的人群。</p>
'''
page+=table(['方法','K=0','K=5','K=10'],realrows)
page+=f'''<figure><img src="real_bridge.svg" alt="真实交互中冻结 Astra 政策落后于内容 BP，增加支持记录改善政策排序"><figcaption class="caption">这里的 NDCG 使用真实正交互的二元相关性；前面是合成效用的分级相关性，绝对数值不能跨任务比较。四个 Astra 网络均与固定编译器相同。</figcaption></figure><p>K=10 时，Astra 相对 BP 三种子均值的差为 {interval['mean_delta']:.5f}，条件配对用户 95% 区间为 [{interval['lower95']:.5f}, {interval['upper95']:.5f}]。相对连续题材余弦的数值增量为 0.01180，但区间跨零；没有可靠优于该内容基线的证据。</p><p>真实 BP 用暖交互学习 BPR 排序，冻结政策没有按真实用户校准，四个题材字段也很有限。因此，这一结果说明本轮固定规则没有迁移出额外预测优势；它不能概括所有 WSP 方法，也不能证明文本无用。v7 的 Llama 电影简介实验与这轮模型、目标及划分不同，不能把两轮差异归因于单一因素。</p>
<h2>07 / 一个可亲手检查的小例子</h2><p>下面直接运行已封存的权重。w05 是任务一的从零写入；r01 和其 BP 来源属于含纪录片例外的任务二。改变“训练时出现过”状态，观察原 BP 与重写网络的差别。演示使用浏览器双精度，不生成新实验、不参与模型选择。</p>
<div class="controls"><fieldset><legend>用户资料</legend><label><input type="checkbox" id="like" checked> 喜欢科幻</label><label><input type="checkbox" id="avoid" checked> 避开露骨内容</label><label><input type="checkbox" id="permit" checked> 允许纪录片例外</label><label><input type="checkbox" id="ustatus" checked> 用户在 BP 训练中出现过</label><label>时间预算 <input type="range" id="budget" min="0.2" max="1" step="0.05" value="0.6"></label></fieldset><fieldset><legend>电影资料</legend><label><input type="checkbox" id="scifi" checked> 科幻</label><label><input type="checkbox" id="graphic" checked> 含露骨内容</label><label><input type="checkbox" id="doc" checked> 纪录片</label><label><input type="checkbox" id="istatus" checked> 电影在 BP 训练中出现过</label><label>片长比例 <input type="range" id="runtime" min="0.1" max="1" step="0.05" value="0.8"></label></fieldset></div><div class="outputs" id="scores"></div><p class="caption">分数越高越优先；单对分数不是推荐指标。w05 的规则始终禁止露骨内容，r01 的规则允许纪录片例外，两个分数目标不同。</p>
<h2>08 / 成本、复核与研究判断</h2>'''
page+=f'''<p>研究生成共 24 个独立上下文、36 次原生 gpt-6-astra 请求；实际工具事件审计为零工具调用。12 次写入各零前向，12 次修复各已有 1 批 64 对混合诊断样例，其中 6 次再用 2 批各 128 对混合反馈；共 24 个诊断／候选批次。后续测试和证明不算编辑器反馈。</p><p>CLI 暴露的研究调用计数：输入 {research_tokens['input_tokens']:,} tokens，其中缓存输入 {research_tokens['cached_input_tokens']:,}；输出 {research_tokens['output_tokens']:,}，另有 reasoning 字段 {research_tokens['reasoning_output_tokens']:,}。字段按原样分别报告，不叠加成账单。没有使用 API key；订阅逐请求金额不可见，美元费用记为未知。外围审计、实现与独立审查不计入这 36 次研究请求；整个目标计数另存成本记录，可能存在计数范围重叠。</p><p>33 次 BP 训练在 A100 上四路并行完成，墙钟约 165 秒，训练试验墙钟之和 {cost['training_wall_seconds_sum']:.1f} 秒。每次训练和编辑设 3,500 秒上限。三次已完成回复含网络重连提示，审计最初误记失败；修正只重用原文，并继续原预算未运行轮次，没有重发已有请求或重试科学内容，原事件及旧记录均保留。</p><p>输入传输的 37 个哈希全部一致；24 个最终权重均封存并逐条重放，测试之前锁定代码与分析。独立测试 91 项通过，v6 的 1,014 个、v7 的 1,172 个旧封存文件未改变。三种任务世界、三个 BP 种子、每格两次编辑不能当成多个独立真实数据集；用户区间只反映固定模型下的用户抽样不确定性。</p><div class="verdict"><strong>这次可支撑的 insight</strong><p>WSP 在推荐中具备“把可理解的偏好政策编译成可执行神经网络”的能力，且能处理新实体组合。这个能力依赖可识别的任务、明确的语义坐标、足够的架构容量与可用信息。未知身份本身不是障碍，未知私人偏好才是信息边界。</p><p>当前证据尚不支撑新的推荐算法、Astra 独有能力、局部机制修复或真实预测优于经典方法。论文可以围绕能力验证与归因边界展开；若主张方法创新，仍需另有能保持已有学习知识、完成政策变化且经典编译器无法直接替代的证据。</p></div>
<details><summary>24 个最终网络的完整逐次结果</summary>'''
page+=table(['编号','实验','条件','双方冷 NDCG','最大数值效用误差','改动参数'],allrows)
page+='''</details><p class="links"><a href="summary.json">汇总</a> · <a href="per_case.csv">逐次 CSV</a> · <a href="baselines.csv">全部 BP 与经典对照</a> · <a href="real_bridge.csv">真实数据全部结果</a> · <a href="paired_user_intervals.json">配对区间</a> · <a href="formal_certificates.json">数学证书</a> · <a href="result_review.json">独立 Astra 审查</a> · <a href="costs.json">成本与用量</a> · <a href="evidence.zip">可核查证据包</a> · <a href="../../paper/v8_mit/main.pdf">更新短文草稿</a></p><footer>WSP4Rec 实验记录 · 结果不服从预设结论。MIT 示例由原作者发布，本研究为独立新实验；MovieLens 数据来自 GroupLens。旧版论文与实验保持为历史版本。</footer></main>'''
page+='<script>const models='+json.dumps(models,separators=(',',':'))+''';function run(s,x){let layer=(a,W,b,relu)=>b.map((v,j)=>{let q=v+a.reduce((n,z,i)=>n+z*W[i][j],0);return relu?Math.max(0,q):q});return layer(layer(layer(x,s.W1,s.b1,true),s.W2,s.b2,true),s.W3,s.b3,false)[0]}function update(){let x=Array(22).fill(0),get=id=>document.getElementById(id).checked?1:0;x[0]=get('like');x[4]=get('scifi');x[10]=get('avoid');x[11]=get('graphic');x[12]=get('permit');x[13]=get('doc');x[14]=+document.getElementById('budget').value;x[15]=+document.getElementById('runtime').value;x[20]=get('ustatus');x[21]=get('istatus');document.getElementById('scores').innerHTML=Object.entries(models).map(([name,s])=>'<span>'+({'w05':'从零写入 w05','r01':'重写后 r01','BP':'原 BP，seed44'}[name])+'<b>'+run(s,x).toFixed(3)+'</b></span>').join('')}document.querySelectorAll('input').forEach(e=>e.addEventListener('input',update));update();</script></html>'''
(R/'report.html').write_text(page)
(R/'formal_certificates.json').write_text(json.dumps(cert,indent=2)+'\n')
print('Built',R/'report.html')
