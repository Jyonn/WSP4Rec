"""Build a Chinese research report and public aggregate evidence, no evaluation.

Requires numpy, reportlab, pypdf. Reads only completed study artifacts.
"""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/study'
REPORT = ROOT / 'reports'
PDF = ROOT / 'output/pdf/WSP4Rec_research_report.pdf'
RESULT = json.loads((RUN / 'final_results.json').read_text())
MANIFEST = json.loads((RUN / 'frozen_manifest.json').read_text())
TRAINING = json.loads((RUN / 'training_summary.json').read_text())
METHODS = ('base', 'weights', 'blind', 'weights_text', 'blind_text', 'calibration', 'random', 'semantic')
NAMES = {'base':'原模型', 'weights':'Astra 读权重', 'blind':'Astra 黑盒', 'weights_text':'Astra 读权重＋标题',
         'blind_text':'Astra 黑盒＋标题', 'calibration':'预设校准', 'random':'预设随机变换', 'semantic':'预设标题平滑'}
SHORT = {'base':'Base', 'weights':'WSP', 'blind':'Blind', 'weights_text':'WSP+T',
         'blind_text':'Blind+T', 'calibration':'Calibrate', 'random':'Random', 'semantic':'TitleSmooth'}
SEEDS = ('42','43','44')
BLUE = colors.HexColor('#173d5e')
TEAL = colors.HexColor('#168987')
GRAY = colors.HexColor('#52616c')
PALE = colors.HexColor('#edf3f6')
RED = colors.HexColor('#ad4d46')


def metric(obj, method, key='ndcg@10'):
    return np.array([RESULT['objectives'][obj][s][method]['metrics']['summary'][key] for s in SEEDS])


def mean(obj, method, key='ndcg@10'):
    return float(metric(obj, method, key).mean())


def fmt(value, n=5):
    return 'N/A' if value is None else f'{value:.{n}f}'


def ms(obj, method, key='ndcg@10'):
    a = metric(obj, method, key)
    return f'{a.mean():.5f} ± {a.std(ddof=1):.5f}'


def delta_group(obj, method, ug, ig):
    return float(np.mean([RESULT['objectives'][obj][s][method]['metrics']['user_item_groups'][ug][ig]['recall@10']
           - RESULT['objectives'][obj][s]['base']['metrics']['user_item_groups'][ug][ig]['recall@10'] for s in SEEDS]))


def export_evidence():
    (REPORT/'tables').mkdir(parents=True, exist_ok=True)
    (REPORT/'evidence').mkdir(parents=True, exist_ok=True)
    rows, group_rows, user_rows, comparisons = [], [], [], []
    aggregates = {}
    for obj in ('mse','bpr'):
        aggregates[obj] = {}
        for method in METHODS:
            aggregates[obj][method] = {k:{'mean':float(metric(obj,method,k).mean()),'sample_sd':float(metric(obj,method,k).std(ddof=1))}
                for k in ('ndcg@10','recall@10','hit_rate@10')}
        for s in SEEDS:
            for method, record in RESULT['objectives'][obj][s].items():
                rows.append({'objective':obj,'seed':s,'method':method,**record['metrics']['summary'],**record['edit_stats']})
                if method != 'base':
                    comparisons.append({'objective':obj,'seed':s,'method':method,**record['paired_bootstrap_vs_base']})
                for ug, item_groups in record['metrics']['user_item_groups'].items():
                    for ig, val in item_groups.items():
                        group_rows.append({'objective':obj,'seed':s,'method':method,'user_group':ug,'item_group':ig,**val})
                for ug, val in record['metrics']['groups'].items():
                    user_rows.append({'objective':obj,'seed':s,'method':method,'user_group':ug,**val})
    for filename, records in (('test_results.csv',rows),('user_item_groups.csv',group_rows),('user_groups.csv',user_rows)):
        with (REPORT/'tables'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    # No raw ratings, titles, vectors or private server details are published.
    for name in ('frozen_manifest.json','training_summary.json'):
        shutil.copy2(RUN/name,REPORT/'evidence'/name)
    shutil.copy2(RUN/'text/title_embeddings.metadata.json',REPORT/'evidence/title_embeddings.metadata.json')
    editor_evidence = {}
    for folder in sorted((RUN/'editors').iterdir()):
        if folder.is_dir():
            editor_evidence[folder.name] = {p.name:json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))}
    (REPORT/'evidence/editor_trajectories.json').write_text(json.dumps(editor_evidence,indent=2,ensure_ascii=False))
    controls = {p.name:json.loads(p.read_text()) for p in sorted((RUN/'controls').glob('*.validation.json'))}
    (REPORT/'evidence/controls.json').write_text(json.dumps(controls,indent=2))
    history_path = RUN/'control_execution_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else list(controls.values())
    (REPORT/'evidence/control_execution_history.json').write_text(json.dumps(history,indent=2))
    costs = {'training_trials':len(TRAINING['trials']), 'training_wall_seconds':sum(t['wall_seconds'] for t in TRAINING['trials']),
       'editor_model':'gpt-6-astra','editor_conditions':8,'editor_generation_turns':16,'editor_candidate_feedback_queries':64,
       'editor_failed_candidates':sum(r['error'] is not None for d in editor_evidence.values() for k,v in d.items() if k.endswith('.validation.json') for r in v['results']),
       'editor_recorded_candidate_execution_seconds':sum(v['wall_seconds'] for d in editor_evidence.values() for k,v in d.items() if k.endswith('.validation.json')),
       'editor_total_inference_seconds':None,'editor_input_tokens':None,'editor_cached_tokens':None,'editor_output_tokens':None,'editor_billed_usd':None,
       'predefined_control_unique_queries':sum(len(d['results']) for d in controls.values()),
       'predefined_control_additional_repeated_queries':max(0,sum(len(d['results']) for d in history)-sum(len(d['results']) for d in controls.values())),
       'predefined_control_latest_execution_seconds':sum(d['wall_seconds'] for d in controls.values()),
       'predefined_control_all_recorded_execution_seconds':sum(d['wall_seconds'] for d in history),
       'references_wall_seconds':json.loads((RUN/'references/validation.json').read_text())['wall_seconds'],
       'final_reporting_wall_seconds':RESULT['wall_seconds'],
       'qwen':json.loads((RUN/'text/title_embeddings.metadata.json').read_text()),
       'compute_timing_scope':'inside-operation wall time; excludes Python startup, network transfers, setup and research-development/coordination model turns; both physical control runs recovered from process output logs',
       'quota_vs_billing':'Subscription use not assigned zero cost; per-call account billing unavailable.',
       'budget':'3600 seconds per experiment. All measured compute operations below limit; exact total inference turn wall unavailable.'}
    (REPORT/'costs.json').write_text(json.dumps(costs,indent=2))
    summary = {'aggregates':aggregates,'references':{k:v['summary'] for k,v in RESULT['references'].items()},
       'paired_user_bootstraps':comparisons,'manifest_sha256':RESULT['manifest_sha256'],
       'title_weight_access_interaction':{o:(mean(o,'weights_text')-mean(o,'blind_text'))-(mean(o,'weights')-mean(o,'blind')) for o in ('mse','bpr')},
       'costs':costs,'scope':'single dataset/split, seed42 recipe discovery + transfer; not independent editor replications'}
    (REPORT/'study_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
STYLES = {
 'title':ParagraphStyle('title',fontName='STSong-Light',fontSize=26,leading=34,textColor=BLUE,spaceAfter=14),
 'sub':ParagraphStyle('sub',fontName='STSong-Light',fontSize=12,leading=18,textColor=GRAY,spaceAfter=10),
 'h1':ParagraphStyle('h1',fontName='STSong-Light',fontSize=17,leading=24,textColor=BLUE,spaceAfter=12),
 'h2':ParagraphStyle('h2',fontName='STSong-Light',fontSize=11.5,leading=17,textColor=TEAL,spaceBefore=8,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='STSong-Light',fontSize=10,leading=16,wordWrap='CJK',spaceAfter=8),
 'small':ParagraphStyle('small',fontName='STSong-Light',fontSize=8.2,leading=12,wordWrap='CJK',spaceAfter=6,textColor=GRAY),
 'cell':ParagraphStyle('cell',fontName='STSong-Light',fontSize=8.4,leading=12,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='STSong-Light',fontSize=8.4,leading=12,textColor=colors.white,wordWrap='CJK'),
}
STORY=[]


def p(text, style='body'):
    return Paragraph(escape(str(text)).replace('\n','<br/>'),STYLES[style])


def add(text, style='body'):
    STORY.append(p(text,style))


def table(headers, rows, widths):
    cells = [[p(h,'head') for h in headers]]+[[p(x,'cell') for x in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#b9cbd5'))]))
    STORY.append(t); STORY.append(Spacer(1,8))


def page(title):
    STORY.append(PageBreak()); add(title,'h1')


def main_figure():
    d=Drawing(490,255)
    for panel,obj in enumerate(('mse','bpr')):
        left=70+panel*245; top=222; width=150; maximum=.135
        d.add(String(left,242,f'{obj.upper()}-MF: test NDCG@10',fontName='Helvetica-Bold',fontSize=10,fillColor=BLUE))
        for j,m in enumerate(METHODS):
            y=top-j*25; values=metric(obj,m); avg=float(values.mean()); sd=float(values.std(ddof=1))
            d.add(String(left-6,y+2,SHORT[m],textAnchor='end',fontName='Helvetica',fontSize=8))
            shade=TEAL if m in ('weights','weights_text') else colors.HexColor('#9aafbe')
            d.add(Rect(left,y,width*avg/maximum,12,fillColor=shade,strokeColor=None))
            x=left+width*avg/maximum; length=width*sd/maximum
            d.add(Line(x-length,y+6,x+length,y+6,strokeColor=BLUE,strokeWidth=.7))
            d.add(String(x+length+3,y+2,f'{avg:.4f}',fontSize=7,fontName='Helvetica',fillColor=BLUE))
        for tick in (0,.05,.10):
            x=left+width*tick/maximum
            d.add(Line(x,28,x,32,strokeColor=GRAY));d.add(String(x,16,f'{tick:.2f}',textAnchor='middle',fontSize=8,fontName='Helvetica'))
        d.add(Line(left,32,left+width,32,strokeColor=GRAY))
    return d


def heatmap():
    d=Drawing(490,300)
    all_values=[abs(delta_group(o,m,u,i)*100) for o in ('mse','bpr') for m in ('weights','weights_text') for u in ('low','mid','high') for i in ('low','mid','high')]
    bound=max(all_values)
    for ix,(obj,method) in enumerate((('mse','weights'),('mse','weights_text'),('bpr','weights'),('bpr','weights_text'))):
        x0=52+(ix%2)*248; y0=197-(ix//2)*146
        d.add(String(x0,y0+80,f'{obj.upper()} {SHORT[method]}: delta Recall@10 (pp)',fontName='Helvetica-Bold',fontSize=9,fillColor=BLUE))
        for r,u in enumerate(('low','mid','high')):
            y=y0+42-r*23
            d.add(String(x0-5,y+7,u,textAnchor='end',fontName='Helvetica',fontSize=8))
            for c,i in enumerate(('low','mid','high')):
                val=delta_group(obj,method,u,i)*100; ratio=abs(val)/bound
                base=TEAL if val>=0 else RED
                shade=colors.Color(1-(1-base.red)*ratio*.85,1-(1-base.green)*ratio*.85,1-(1-base.blue)*ratio*.85)
                d.add(Rect(x0+c*54,y,53,22,fillColor=shade,strokeColor=colors.white))
                d.add(String(x0+c*54+26,y+7,f'{val:+.2f}',textAnchor='middle',fontName='Helvetica',fontSize=8,fillColor=BLUE))
        for c,i in enumerate(('low','mid','high')):
            d.add(String(x0+c*54+26,y0-17,i,textAnchor='middle',fontName='Helvetica',fontSize=8))
    return d


def budget_figure():
    d=Drawing(490,172)
    shades=[BLUE,colors.HexColor('#b77435'),TEAL,colors.HexColor('#955c8d')]
    for panel,obj in enumerate(('mse','bpr')):
        x0=38+panel*245; y0=35; width=185; height=105
        sequences=[MANIFEST['choices'][obj][m]['valid_curve_best_so_far'] for m in ('weights','blind','weights_text','blind_text')]
        lo=min(min(s) for s in sequences)-.001; hi=max(max(s) for s in sequences)+.001
        d.add(String(x0,155,obj.upper()+' validation best-so-far',fontName='Helvetica-Bold',fontSize=9,fillColor=BLUE))
        for seq,col in zip(sequences,shades):
            pts=[(x0+j*width/7,y0+(v-lo)/(hi-lo)*height) for j,v in enumerate(seq)]
            for a,b in zip(pts,pts[1:]):d.add(Line(*a,*b,strokeColor=col,strokeWidth=1.5))
            for x,y in pts:d.add(Circle(x,y,1.8,fillColor=col,strokeColor=None))
        for j in (0,3,7):d.add(String(x0+j*width/7,y0-13,str(j+1),textAnchor='middle',fontSize=8,fontName='Helvetica'))
        d.add(String(x0-4,y0,f'{lo:.3f}',textAnchor='end',fontSize=7,fontName='Helvetica'));d.add(String(x0-4,y0+height,f'{hi:.3f}',textAnchor='end',fontSize=7,fontName='Helvetica'))
        d.add(Line(x0,y0,x0+width,y0,strokeColor=GRAY))
    for j,(label,col) in enumerate(zip(('WSP','Blind','WSP+T','Blind+T'),shades)):
        d.add(Line(70+j*105,6,85+j*105,6,strokeColor=col,strokeWidth=2));d.add(String(90+j*105,3,label,fontSize=8,fontName='Helvetica'))
    return d


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#d5e0e6'));canvas.line(20*mm,17*mm,190*mm,17*mm)
    canvas.setFont('STSong-Light',8);canvas.setFillColor(GRAY)
    canvas.drawString(20*mm,12*mm,'WSP4Rec | MovieLens-100K | 2026-11-04')
    canvas.drawRightString(190*mm,12*mm,str(doc.page))


def build(summary):
    cost=summary['costs']; names=RESULT['objectives']['bpr']['42']['base']['metrics']['grouping']
    base=mean('bpr','base'); wsp=mean('bpr','weights'); wt=mean('bpr','weights_text')
    add('WSP4Rec 研究报告','title')
    add('直接权重编辑在协同推荐中的有效性、分布性影响与标题信息作用','sub')
    add('研究日期：2026 年 11 月 4 日｜阶段：首轮受控实证研究｜面向 WWW 短文选题决策','small')
    add('主要发现','h2')
    add(f'在 MovieLens-100K 的固定离线协议中，Astra 可以通过无梯度权重变换改善推荐，但目前没有建立“读取权重具有独特优势”的证据。MSE-MF 的平均测试 NDCG@10 从 {mean("mse","base"):.5f} 提高到 {mean("mse","weights"):.5f}；黑盒编辑与简单校准也能取得类似或更好的结果。')
    add(f'在更强的 BPR-MF 上，无标题 WSP 从 {base:.5f} 变为 {wsp:.5f}（相对变化 {(wsp/base-1)*100:+.2f}%），且三种子中有一个退化。标题辅助 WSP 为 {wt:.5f}，但预设标题平滑为 {mean("bpr","semantic"):.5f}。结果支持继续研究编辑的信息条件，不支持宣布一种普遍优于常规推荐训练的新方法。')
    table(['研究问题','本轮回答'],[
       ['RQ1 无文本是否有效','弱 MSE 基线可明显改善；强 BPR 基线收益小且不稳定。'],
       ['RQ2 谁获益、谁受损','收益依赖用户活跃度与物品支持度；没有观察到低支持物品 Recall@10 的改善。'],
       ['RQ3 标题是否有用','BPR 中存在小幅信号，且正确对应的表示优于打乱对应；尚未优于预设语义平滑。'],
       ['RQ4 权重可见性是否必要','无标题条件下没有显示额外优势；标题条件的交互项为正，但仅有一次编辑发现轨迹。']
    ],[132,350])
    add('建议定位：一项关于推荐权重编辑的可行性与归因研究。当前更有价值的发现是“常规校准能解释多少收益、标题邻居何时有用”，而非性能领先或通用冷启动能力。','small')

    page('实验协议与信息边界')
    add('固定基础推荐器与评估口径','h2')
    add('943 名用户、1,682 部电影、100,000 条评分。每用户按时间排序，训练／验证／测试为 79,619／9,942／10,439 条；同时间戳按物品 ID 排序。这是每用户时间划分，不能解释为全局时间上的线上预测。物品目录来自固定 u.item 元数据。')
    add('评分≥4 作为正例。主指标为全目录 NDCG@10，辅指标 Recall@10、HitRate@10。验证排除所有训练交互，测试排除训练及验证交互；未观察物品作为候选不等于其真实不相关。测试共有 856 名用户有 eligible 正例，共 4,975 个正例。')
    add('模型是两字段 FM，等价于带偏置 MF，向量维度 16，共 44,626 个参数。MSE 与 BPR 分别调参。种子 42 上搜索学习率 0.003／0.01、weight decay 1e-5／0.001；上限 80 epoch、12 epoch 耐心值，验证 NDCG@10 选 checkpoint。再用选定超参数训练种子 43、44。BPR 正例为高评分，负例均匀采自未训练交互物品；低评分不直接作为负例。')
    add('MSE 三个种子的最优 checkpoint 均在第 3 epoch，运行至第 15 epoch 早停。因此这里只能称“按验证排序选出的 MSE 模型”，不能宣称已经充分收敛的评分预测解仍被显著改进。','small')
    table(['条件','编辑者可见信息','相同反馈预算'],[
       ['WSP','结构、训练计数、真实参数','2 轮 × 每轮 4 候选'],
       ['Blind','结构、训练计数；不可看真实参数','同上'],
       ['WSP＋标题','WSP 信息＋标题与 Qwen 表示','同上'],
       ['Blind＋标题','Blind 信息＋标题与 Qwen 表示','同上']
    ],[90,270,122])
    add('每种训练目标的四个编辑条件使用独立 Astra 子代理。仅种子 42 用于发现与选择 recipe，之后固定迁移到种子 43、44，不再选择。所有候选从同一原始 checkpoint 出发；原模型始终可选。编辑使用缩放、支持度修正、范数调整、维度缩放或邻居平滑等代数操作，不进行梯度更新或数据拟合。')
    add('Blind 与 WSP 共用相对编辑接口，运行器可在内部使用私有权重。比较的是“编辑者能否观察权重”，不是“整个过程是否利用权重”。文本条件同时开放标题输入与基于 Qwen 的邻居算子，因此研究的是标题辅助流程的联合效应。')
    add('原始交互与测试标签不提供给编辑者；反馈仅包含自己的候选 ID 和验证分数，编辑范数保留在审计日志。子代理遵守显式文件白名单，但没有建立容器级文件系统隔离。全部选择、checkpoint、标题表示和代码哈希冻结后，统一进行一次测试报告。','small')

    page('总体结果与基线强度')
    add('表 1  三种子测试均值 ± 样本标准差；同一数据划分，编辑策略并非独立重复。','small')
    table(['方法','MSE NDCG@10','BPR NDCG@10','MSE Recall','BPR Recall'],[
       [NAMES[m],ms('mse',m),ms('bpr',m),fmt(mean('mse',m,'recall@10')),fmt(mean('bpr',m,'recall@10'))] for m in METHODS
    ],[114,113,113,71,71])
    fig=main_figure();STORY.append(fig);renderSVG.drawToFile(fig,str(REPORT/'figures/test_ndcg.svg'))
    add('图 1  条为平均测试 NDCG@10，误差线为三种子标准差；不是置信区间。','small')
    ease=RESULT['references']['ease']['summary']['ndcg@10']; pop=RESULT['references']['popularity']['summary']['ndcg@10']
    add(f'上下文参照：热门推荐 NDCG@10={pop:.5f}；验证选择的 EASE={ease:.5f}。EASE 使用训练正例矩阵和四个正则化候选，不是同信息／同预算的编辑控制，也没有随机种子标准差。它说明本轮编辑后的 MF 仍未达到这一传统强参照。')
    add('无标题 BPR WSP 最终选择仅将物品偏置乘 0.5，和预设校准的选择完全一致，因此两者在三个种子上的测试结果逐一相同。不能将这个结果解释为 LLM 独有的权重推理能力。','small')

    page('RQ2 交互程度与连带损害')
    add('用户训练交互 tertile 阈值为 33、91；物品正训练支持 tertile 阈值为 8、43，零支持另列，共 71 个物品。计数并列使各桶大小不完全相等。这里的“冷”指低交互支持，不等于零历史冷启动。')
    fig=heatmap();fig.scale(.85,.85);fig.width*=.85;fig.height*=.85;fig.hAlign='CENTER';STORY.append(fig);renderSVG.drawToFile(fig,str(REPORT/'figures/group_deltas.svg'))
    add('图 2  编辑相对原模型的 Recall@10 变化，单位百分点，平均于三个种子。行是用户活跃度，列是物品支持度；每格仍从全目录 Top10 计算，未在组内重排。','small')
    table(['用户组','低支持物品 用户／正例','中支持物品 用户／正例','高支持物品 用户／正例'],[
      [ug]+[str(RESULT['objectives']['bpr']['42']['base']['metrics']['user_item_groups'][ug][ig]['ranking_users'])+' / '+str(RESULT['objectives']['bpr']['42']['base']['metrics']['user_item_groups'][ug][ig]['positives']) for ig in ('low','mid','high')] for ug in ('low','mid','high')
    ],[65,139,139,139])
    add('在 WSP 与 WSP＋标题的最终策略中，低支持物品的 Recall@10 均没有改善。MSE 的可见提升集中在高支持物品；BPR 的变化较小并混合正负。低支持格的正例与用户数很少，因此结果不能用来证明 WSP 对所有长尾问题无效。')
    table(['模型／编辑','平均改善用户比例','平均退化用户比例'],[
      [o.upper()+' / '+SHORT[m],f'{np.mean([RESULT["objectives"][o][s][m]["paired_bootstrap_vs_base"]["improved_fraction"] for s in SEEDS])*100:.1f}%',f'{np.mean([RESULT["objectives"][o][s][m]["paired_bootstrap_vs_base"]["harmed_fraction"] for s in SEEDS])*100:.1f}%'] for o in ('mse','bpr') for m in ('weights','weights_text')
    ],[200,141,141])
    add('改善／退化按用户 NDCG@10 的严格正负变化统计；其余用户保持不变。曝光与覆盖率也只在这 856 名有 eligible 正例的用户上计算，不是全用户公平性或线上体验结论。','small')

    page('RQ3 与 RQ4 标题作用和权重访问价值')
    add('文本特征固定为 Qwen3-Embedding-0.6B 对原始电影标题（保留年份）的 1,024 维末 token 表示，L2 归一化。它们不进入基础模型训练和在线推理，而用于编辑阶段的物品邻居平滑。标题相似性可能混合名称、年份、实体知识及预训练记忆，不能等同于纯内容语义。')
    table(['三种子 NDCG 差值','MSE','BPR'],[
       ['无标题的权重可见收益：WSP - Blind',fmt(mean('mse','weights')-mean('mse','blind')),fmt(mean('bpr','weights')-mean('bpr','blind'))],
       ['有标题的权重可见收益：WSP+T - Blind+T',fmt(mean('mse','weights_text')-mean('mse','blind_text')),fmt(mean('bpr','weights_text')-mean('bpr','blind_text'))],
       ['上述两项之差：标题 × 权重访问交互',fmt(summary['title_weight_access_interaction']['mse']),fmt(summary['title_weight_access_interaction']['bpr'])],
       ['WSP+T - 预设标题平滑',fmt(mean('mse','weights_text')-mean('mse','semantic')),fmt(mean('bpr','weights_text')-mean('bpr','semantic'))],
       ['WSP+T：正确表示对应 - 打乱对应',fmt(mean('mse','weights_text')-mean('mse','weights_text_shuffled')),fmt(mean('bpr','weights_text')-mean('bpr','weights_text_shuffled'))]
    ],[322,80,80])
    add('BPR 标题辅助 WSP 选择“偏置乘 0.5＋全物品 15% 的五邻居文本平滑”。正确表示对应优于固定打乱对应，支持这一编辑对文本邻居对应有依赖；但单独预设标题平滑的平均测试分数更高。MSE 中打乱低支持邻居并不改变 Top10 分数，说明最终收益主要可由全局校准解释。')
    add('这里的打乱诊断仅替换编译器中的物品-表示对应，保持选定 recipe，不重新向编辑者提供乱标题，也没有再选择。它不能证明编辑者理解了电影语义。BPR 交互逐种子为 +0.00277、-0.00049、-0.00032，正均值主要来自发现种子 42。每条件仅一条发现轨迹，无编辑模型随机性置信区间。')
    add('固定种子下的配对不确定性','h2')
    rows=[]
    for o in ('mse','bpr'):
        for m in ('weights','weights_text'):
            b=RESULT['objectives'][o]['42'][m]['paired_bootstrap_vs_base']
            rows.append([o.upper()+' / '+SHORT[m],fmt(b['mean_delta']),f'[{b["ci95"][0]:.5f}, {b["ci95"][1]:.5f}]'])
    table(['seed42 对原模型','ΔNDCG@10','配对用户 bootstrap 95% 区间'],rows,[175,105,202])
    add('2,000 次配对用户 bootstrap 仅条件于固定训练模型和划分，未校正多重比较；不能替代独立训练或编辑重复。BPR 无标题区间跨零，标题条件仅有单种子的边缘信号，不能据此写成稳定显著提升。','small')

    page('反馈预算与成本')
    fig=budget_figure();STORY.append(fig);renderSVG.drawToFile(fig,str(REPORT/'figures/validation_budget.svg'))
    add('图 3  seed42 验证集的当前最优曲线，包含原模型回退。每轮四个候选一次性提出，只有第 4 次后向编辑者回传反馈；曲线不代表每次查询后都自适应。','small')
    q=cost['qwen']
    table(['资源或事件','本轮测量'],[
       ['基础模型训练','12 个试验；内部墙钟合计 '+f'{cost["training_wall_seconds"]:.2f} 秒；CPU、单线程'],
       ['Astra 正式编辑','8 条轨迹、16 次生成回合、64 个候选反馈；失败候选 0'],
       ['候选编译与验证','64 个候选的记录墙钟合计 '+f'{cost["editor_recorded_candidate_execution_seconds"]:.2f} 秒'],
       ['预设编辑对照',f'{cost["predefined_control_unique_queries"]} 个唯一候选；额外重复执行 {cost["predefined_control_additional_repeated_queries"]} 次；不是额外搜索预算'],
       ['Qwen 标题提取',f'单 A100；墙钟 {q["wall_seconds"]:.2f} 秒（载入 {q["load_seconds"]:.2f}、编码 {q["encode_seconds"]:.2f}）；约 {q["peak_gpu_memory_bytes"]/1e9:.2f} GB 峰值显存'],
       ['Qwen 输入','1,682 个标题；19,396 个有效输入 token；一次性表示构建成本'],
       ['最终测试报告',f'{RESULT["wall_seconds"]:.2f} 秒；统一已冻结策略＋固定打乱诊断'],
       ['无法精确取得的字段','Astra 输入／缓存／输出 token、完整推理时长、逐次订阅金额；均记为 null']
    ],[145,337])
    add(f'墙钟计时在操作函数内完成，不含 Python 导入、SSH、下载、传输和环境准备；不能称为端到端总成本。部分文本编辑回合另有首个工具动作到完成的观察时长，仅为局部时间。对照执行记录含重复计算，{cost["predefined_control_unique_queries"]+cost["predefined_control_additional_repeated_queries"]} 次物理评估合计 {cost["predefined_control_all_recorded_execution_seconds"]:.2f} 秒。研究开发与独立审查对话不摊入正式编辑成本。')
    add('已测计算任务均远低于单实验 1 小时上限；Astra 的完整单回合推理时长未由接口暴露。原生子代理使用当前订阅，没有可用逐次账单，不能写成零成本或与其他模型等成本。模型调用与 CPU/GPU 计算分别记录，未将订阅月费任意摊销。','small')

    page('证据边界与 WWW 短文建议')
    add('当前可以写入论文的结论','h2')
    table(['主张','证据与状态'],[
       ['无梯度编辑能改变离线推荐表现','支持，但限定为本数据、MF、操作接口和预算；MSE 改善更明显。'],
       ['LLM 读取权重优于黑盒或简单校准','未建立。无标题条件权重访问未优于黑盒，BPR 选择与预设校准一致。'],
       ['交互稀疏群体更受益','不支持普遍表述。低支持物品未见 Recall@10 改善，用户和中高支持物品变化异质。'],
       ['标题能拓展 WSP 能力','有探索性信号，不能归因于 LLM 语义理解或宣称稳定优势。'],
       ['方法低成本、可扩展到工业规模','未验证；目前只有小数据、结构化变换与不完整的模型用量数据。']
    ],[182,300])
    add('限制','h2')
    add('仅一个数据集、一个划分、16 维 MF；seed42 同时用于调参和编辑发现。三个训练种子不是三个独立编辑实验；没有因果操纵交互密度。随机控制搜索空间比 Astra 窄，预设控制不自适应，因此不能声称完整预算／搜索空间匹配。编辑算子预先规定，结果不是对任意权重编程能力的穷尽检验。')
    add('纯 ID 条件没有提供标题，但也没有对物品编号重新匿名排列，公开数据可能存在于 LLM 预训练中。零支持物品可在 BPR 中被采作负例，不构成真正新物品冷启动。低支持测试正例较少；离线未观察候选和实际曝光机制也限制了外推。')
    add('短文下一步应围绕一个主张补证据','h2')
    add('建议优先重复“权重可见 vs 黑盒”的独立编辑轨迹，在另一个小型数据集／划分上验证；随后固定校准强度，再比较标题邻居和平滑算子，避免把校准收益混入语义收益。进一步移除标题年份、加入同格式乱配标题的完整编辑条件，以及与编辑空间一致的自适应非 LLM 搜索。这些属于后续研究，未在本轮测试结果上继续调参。')
    add('五维自审','h2')
    add('贡献：归因与边界有价值，方法领先性不足。清晰度：参数、预算、分组和选择可复现。实验强度：有强参照与反例，独立编辑重复不足。评价完整性：四个 RQ 均有对应证据，单数据集与成本缺失仍是限制。方法合理性：编辑与训练分离；算子约束和仅指令隔离须如实说明。','small')

    page('复现与来源')
    add('复现入口','h2')
    add('代码仓库：https://github.com/Jyonn/WSP4Rec\n完整协议：study_protocol.json、study_amendments.json、editor_interface.json\n公开汇总：reports/study_summary.json、reports/tables/、reports/evidence/\n报告生成：scripts/build_study_report.py')
    add('训练与选择代码为 wsprec.study；测试封存和统一报告为 wsprec.finalize_study。原生 Astra 编辑不能用同一随机种子精确重放，故保留了全部 64 个候选、两轮反馈、数值诊断与最终 recipe，以便不重新调用模型时重现变换与结果。生成表格、图和 PDF 的脚本只读取现成结果，不再次评估测试集。')
    add('冻结清单 SHA256：'+RESULT['manifest_sha256'],'small')
    add('Qwen 固定版本：'+cost['qwen']['resolved_revision'],'small')
    add('参考来源','h2')
    sources=[
      '[1] Takehi et al. Sparks of Weight-Space Programming. 2026-11-02. https://rikiyat.github.io/blog/weight-space-programming/',
      '[2] GroupLens. MovieLens 100K Dataset. https://grouplens.org/datasets/movielens/100k/',
      '[3] Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009. https://arxiv.org/abs/1205.2618',
      '[4] Steck. Embarrassingly Shallow Autoencoders for Sparse Data. WWW 2019. https://arxiv.org/abs/1905.03375',
      '[5] Lai et al. Better Late Than Never: Formulating and Benchmarking Recommendation Editing. 2024. https://arxiv.org/abs/2406.04553',
      '[6] Qwen. Qwen3-Embedding-0.6B model card. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B',
      '[7] ACM Web 2027. Research Track Papers / short-paper format. https://www2027.thewebconf.org/research-track-papers/'
    ]
    for s in sources:add(s,'small')
    add('这些来源分别支持原始 WSP 概念、数据来源、训练目标、EASE 方法、推荐编辑相关工作、标题表示实现及投稿格式；本报告所有性能数字来自本地实验，不来自上述来源。未将“推荐模型编辑”本身声称为首次提出。','small')
    PDF.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(PDF),pagesize=(210*mm,297*mm),rightMargin=20*mm,leftMargin=20*mm,topMargin=20*mm,bottomMargin=23*mm,
        title='WSP4Rec 研究报告',author='WSP4Rec Research')
    doc.build(STORY,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    (REPORT/'figures').mkdir(parents=True,exist_ok=True)
    summary=export_evidence();build(summary)
    print(PDF)
