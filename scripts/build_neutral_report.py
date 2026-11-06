"""Render a factual report from the completed neutral study; no evaluation."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer
from reportlab.graphics.shapes import Drawing,Rect,String,Line

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/os.environ.get('WSPREC_RUN_DIR','runs/v3_neutral');REPORT=ROOT/os.environ.get('WSPREC_REPORT_DIR','reports/v3_neutral')
S=json.loads((REPORT/'summary.json').read_text())
R=json.loads((RUN/'final_results.json').read_text())
SEEDS=('42','43','44');TEXTS=('none','title','description')
LABELS={'none':'无文本','title':'仅标题','description':'标题＋简介'}
BLUE=colors.HexColor('#173d5e');TEAL=colors.HexColor('#168987');PALE=colors.HexColor('#edf3f6')
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
STYLES={key:ParagraphStyle(key,fontName='STSong-Light',fontSize=size,leading=leading,textColor=color,
    wordWrap='CJK',spaceAfter=space) for key,size,leading,color,space in
    [('title',23,30,BLUE,14),('h',16,23,BLUE,12),('body',9.5,15,colors.black,8),
     ('small',8,12,colors.HexColor('#52616c'),6),('cell',8.4,12,colors.black,0),('head',8.4,12,colors.white,0)]}
story=[]


def p(text,style='body'):return Paragraph(escape(str(text)).replace('\n','<br/>'),STYLES[style])
def add(text,style='body'):story.append(p(text,style))
def page(title):story.append(PageBreak());add(title,'h')
def table(headers,rows,widths):
    t=Table([[p(v,'head') for v in headers]]+[[p(v,'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,9)])
def avg(obj,text,vis,key='ndcg@10'):return S['aggregates'][obj][f'{text}-{vis}'][key]
def control(obj,family,key='ndcg@10'):return np.mean([R['objectives'][obj][s][f'{obj}-{family}']['metrics']['summary'][key] for s in SEEDS])
def unique(obj,text,vis):return np.mean([R['objectives'][obj][s][f'{obj}-{text}-{vis}-r{rep}']['unique_text_target_summary']['ndcg@10'] for s in SEEDS for rep in (0,1)])


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#ccd9e2'));canvas.line(20*mm,17*mm,190*mm,17*mm)
    canvas.setFont('STSong-Light',8);canvas.setFillColor(BLUE)
    canvas.drawString(20*mm,12*mm,'WSP4Rec | Description complete | 2026-11-06');canvas.drawRightString(190*mm,12*mm,str(doc.page))


def chart():
    d=Drawing(480,185);maximum=max(.01,max(avg(o,t,v) for o in ('mse','bpr') for t in TEXTS for v in ('visible','blind')))*1.2
    for panel,obj in enumerate(('mse','bpr')):
        x0=50+panel*245;y0=138;width=158
        d.add(String(x0,y0+26,obj.upper()+' mean test NDCG@10',fontName='Helvetica-Bold',fontSize=9,fillColor=BLUE))
        for i,text in enumerate(TEXTS):
            y=y0-i*45;d.add(String(x0-5,y-1,('ID','Title','Desc')[i],textAnchor='end',fontName='Helvetica',fontSize=8))
            for j,vis in enumerate(('visible','blind')):
                val=avg(obj,text,vis);ybar=y-j*13;col=TEAL if vis=='visible' else colors.HexColor('#9aafbe')
                d.add(Rect(x0,ybar,width*val/maximum,10,fillColor=col,strokeColor=None))
                d.add(String(x0+width*val/maximum+3,ybar+1,f'{val:.4f}',fontName='Helvetica',fontSize=7))
        d.add(Line(x0,24,x0+width,24,strokeColor=BLUE))
        for tick in (0,maximum/2,maximum):d.add(String(x0+width*tick/maximum,12,f'{tick:.3f}',textAnchor='middle',fontName='Helvetica',fontSize=7))
    return d


def main():
    d=S['dataset'];c=S['costs'];n=d['counts']['after'];splits=d['split_sizes']
    add('WSP4Rec 新数据集实验报告','title')
    add('简介完整的电影子集；无方向性假设提示；通用张量编辑接口。','small')
    add('就本轮测试均值而言，加入简介未超过对应的仅标题条件，读取权重也没有稳定优于盲编辑。这些观察限于本轮策略、数据及预算。')
    add(f'本轮 BPR 原模型平均 NDCG@10 为 {S["aggregates"]["bpr"]["base"]["ndcg@10"]:.5f}；读取权重的无文本编辑为 {avg("bpr","none","visible"):.5f}，仅标题为 {avg("bpr","title","visible"):.5f}，标题＋简介为 {avg("bpr","description","visible"):.5f}。盲编辑的标题＋简介为 {avg("bpr","description","blind"):.5f}。这些是已封存策略的测试结果，不是预设效果或对内部动机的推断。')
    removed=d['removals']['no_reliable_description_match_in_this_source']
    add(f'本轮保留 {n["users"]} 名用户、{n["items"]} 部电影及 {n["interactions"]:,} 条交互。移除 {removed["items"]} 部没有可靠公开简介匹配、简介为空或为占位文本的电影及 {removed["interactions"]:,} 条交互。这个筛选不意味着被移除电影在所有公开来源都不存在简介。')
    add(f'过滤后重新按用户时间顺序划分，训练／验证／测试分别为 {splits["train"]:,}／{splits["valid"]:,}／{splits["test"]:,}。用户和电影重新映射到固定随机排列的匿名 ID；旧数据、旧模型、旧测试结果未被覆盖。')
    table(['事项','本轮实施'],[
        ['编辑提示','不提供改进假设、方向性操作示例或旧实验结论；不要求解释、理由或假设。'],
        ['编辑接口','仅加法、逐元素乘法、替换、左右矩阵乘法；任一实际参数均可修改。'],
        ['候选约束','每个 patch 至多 6 操作；参数绝对值≤100；整体参数 L2 改变量≤原始范数50%。'],
        ['信息条件','权重可见／不可见 × 无文本／仅标题／标题＋简介；两种基础训练目标。'],
        ['重复与预算','每条件两个独立编辑者；每人两轮、每轮四个候选；固定修改再迁移至三个训练种子。'],
        ['反馈','只返回自己的候选标识、验证 NDCG@10 或统一无效状态；越界也消耗候选预算。']
    ],[108,374])
    add('MSE 与 BPR 均为 16 维带偏置矩阵分解，种子42验证选择学习率和正则化，种子43、44按相同设置重训。正例评分≥4；全目录 Top10，验证排除训练交互，测试排除训练和验证交互。全部候选、资产、模型与代码封存哈希后统一报告测试。')
    add('基础推荐模型训练只使用交互；文本和向量仅提供给编辑者。文本表示复用固定 Meta-Llama-3-8B 编码的逐电影缓存，按保留电影与新 ID 取子集。仅标题和标题＋简介来自同一模型及相同任务提示；所有保留电影都有简介，没有本轮缺失回退。该模型未针对 embedding 微调。')
    add('完整提示词保存在 evidence/prompts.json；指令审计不等于证明模型没有预训练先验或内部推理。原生子代理使用新上下文和文件白名单，没有建立操作系统级文件隔离。','small')

    page('测试结果与独立编辑重复')
    table(['条件','MSE NDCG@10','BPR NDCG@10','MSE Recall@10','BPR Recall@10'],
        [['原模型',f'{S["aggregates"]["mse"]["base"]["ndcg@10"]:.5f}',f'{S["aggregates"]["bpr"]["base"]["ndcg@10"]:.5f}',f'{S["aggregates"]["mse"]["base"]["recall@10"]:.5f}',f'{S["aggregates"]["bpr"]["base"]["recall@10"]:.5f}']]+
        [[LABELS[t]+'／'+('读权重' if v=='visible' else '盲编辑'),f'{avg("mse",t,v):.5f}',f'{avg("bpr",t,v):.5f}',f'{avg("mse",t,v,"recall@10"):.5f}',f'{avg("bpr",t,v,"recall@10"):.5f}'] for t in TEXTS for v in ('visible','blind')],[118,91,91,91,91])
    add(f'验证选择保持原模型的编辑过程：{len(S["no_op_trajectories"])}／24；跨训练种子迁移因边界无效而回退：{len(S["transfer_invalid_cases"])} 次。完整条目记录在 summary.json。','small')
    add('表中编辑值平均于两个独立编辑过程及各自的三个训练种子迁移；不是六个独立编辑过程。原模型仅平均三个训练种子。下图青色为读权重，灰色为盲编辑。','small')
    story.append(chart())
    add('图中各条仅描述本轮已冻结策略的实际分数，不表示编辑模型在其他数据集上的总体优势。完整逐种子数据在 test_results.csv。','small')
    page('独立编辑重复与传统对照')
    for obj in ('mse','bpr'):
        table([obj.upper(),'读权重 rep0','读权重 rep1','盲编辑 rep0','盲编辑 rep1'],[[LABELS[t]]+
            [f'{S["aggregates"][obj][t+"-"+v]["editor_rep_means_ndcg"][r]:.5f}' for v in ('visible','blind') for r in (0,1)] for t in TEXTS],[94,97,97,97,97])
    add('每个训练种子的用户配对 bootstrap 区间见 paired_user_intervals.json；区间以固定训练模型及策略为条件，不描述编辑器调用总体。','small')
    add('每格是同一固定编辑策略在三个训练种子上的测试平均。两个编辑重复只提供有限的随机性检查，不足以建立编辑模型总体显著性；不能把同策略的训练种子当作独立模型调用重复。','small')

    table(['预设控制','MSE NDCG@10','BPR NDCG@10'],[[label,f'{control("mse",family):.5f}',f'{control("bpr",family):.5f}'] for family,label in
        (('uniform','所有参数统一数值缩放'),('random','固定种子的随机数值更新'),('title_neighbors','固定标题邻居变换'),('description_neighbors','固定简介邻居变换'))],[250,116,116])
    add(f'基于同一训练集构建的参照：热门推荐 NDCG@10={S["references"]["popular"]["ndcg@10"]:.5f}，EASE={S["references"]["ease"]["ndcg@10"]:.5f}。它们不是预算相同的编辑器。预设控制各八候选，但搜索空间和自适应程度与 Astra 不完全相同。')
    page('分组与数据质量诊断')
    rows=[]
    for obj in ('mse','bpr'):
        base=R['objectives'][obj]['42']['base']['metrics']
        for ug in ('low','mid','high'):
            row=[obj.upper()+'／'+ug]
            for ig in ('low','mid','high'):
                values=[R['objectives'][obj][s][f'{obj}-description-visible-r{rep}']['metrics']['user_item_groups'][ug][ig]['recall@10']-
                        R['objectives'][obj][s]['base']['metrics']['user_item_groups'][ug][ig]['recall@10'] for s in SEEDS for rep in (0,1)]
                row.append(f'{np.mean(values)*100:+.2f}')
            rows.append(row)
    table(['用户活跃组','低支持物品 ΔRecall','中支持物品 ΔRecall','高支持物品 ΔRecall'],rows,[130,117,117,118])
    add('上表为标题＋简介、权重可见编辑相对原模型的变化，单位百分点。分组阈值只依赖新训练集计数，分组评价仍保留全目录竞争。完整格用户数、正例数及所有方法在 user_item_groups.csv，零训练支持物品另列。','small')
    table(['目标电影文本唯一子集','MSE NDCG@10','BPR NDCG@10'],[[LABELS[t]+'／读权重',f'{unique("mse",t,"visible"):.5f}',f'{unique("bpr",t,"visible"):.5f}'] for t in TEXTS],[250,116,116])
    add('元数据核查发现18条重复标题＋简介记录，因此在测试打开前额外固定了“仅评价文本唯一电影目标”的诊断，候选仍为全目录。该诊断不影响训练、编辑反馈或选择，不可与全体目标的绝对分数直接比较。')
    add('当前匹配保留101个唯一名称但年份相差一年的对应；未逐部人工核验。筛选后的电影子集改变了原始流行度分布。仅保留已获取简介并不能消除来源选择偏差。','small')

    page('执行记录与结论边界')
    table(['记录','数量或状态'],[
        ['新基线训练',f'{c["training_trials"]} 次；函数内墙钟合计 {c["training_wall_seconds"]:.2f} 秒'],
        ['独立编辑过程',f'{c["editor_trajectories"]} 条；{c["editor_generation_turns"]} 个完整记录生成回合'],
        ['候选反馈',f'{c["editor_candidate_queries"]} 次；其中 {c["invalid_editor_candidates"]} 次无效，均计入预算'],
        ['预设控制',f'{c["control_candidate_queries"]} 次候选，{c["invalid_control_candidates"]} 次无效'],
        ['编译及验证',f'函数内墙钟合计 {c["candidate_execution_wall_seconds"]:.2f} 秒'],
        ['封存一致性校验',f'{c["integrity_replay_evaluations"]} 次快照重放，{c["integrity_replay_wall_seconds"]:.2f} 秒；不向编辑者提供反馈'],
        ['EASE拟合与验证',f'{c["ease_reference_fit_and_validation_wall_seconds"]:.2f} 秒；不含 Gram 构造与热门参照构造'],
        ['最终测试报告',f'{c["test_reporting_wall_seconds"]:.2f} 秒，包含固定元数据诊断'],
        ['文本构建成本','本轮复用历史缓存，未新增 GPU 编码；历史构建成本另有记录，不是零成本。'],
        ['不可获得字段','原生编辑器 token、完整推理时长和逐调用账单均为 null。']
    ],[160,322])
    if c.get('execution_notes'):add('另有一次中断时未确认执行的派发尝试，单独记录，未增加候选评分；不能据此给出精确调用账单。','small')
    total=c.get('including_intermediate',{})
    if total:add(f'额外计入已废弃的中间轮：因发现两条占位简介，1,570 部电影的实验不作为最终结果。含该轮累计 {total["editor_generation_turns"]} 个生成回合、{total["editor_candidate_queries"]} 个编辑候选、{total["training_trials"]} 次训练。费用无法按回合精确计价。')
    add('数据准备、控制矩阵构建及编辑者的数值处理未完整计时。其余计时也不含 Python 启动、开发、协调、下载与同步，不是端到端总成本。相同候选数不等于相同金钱或计算费用。相对范数限制对盲编辑者的加法更新可行性也有影响；无效次数按条件公开，不能把所有组间差异都归因于推荐理解。')
    add('本轮只报告已观察到的分数与修改，不收集或推断编辑者的内部动机。通用接口仍有参数和幅度限制，不能据此证明任意权重编程能力。训练种子迁移若违反约束，固定回退原模型，不重新选择。')
    add('新数据来自同一 MovieLens 原始语料，和旧划分存在重叠，不是独立外部测试。数据、匿名 ID、接口、提示和文本编码条件均有变化，因此不能把新旧分数差异归因于“去掉假设”这一个因素。该研究只有一个过滤后的电影数据集。')
    add('复现与证据：neutral_protocol_v3.json、editor_interface_v2.json、editor_prompt_v2.txt；reports/v3_neutral/evidence/ 保存全部提示、候选、反馈与冻结清单；numeric_patch_assets.zip 用哈希去重保存数值资产；test_results.csv 和 summary.json 保存完整结果。')
    add('冻结清单 SHA256：'+S['manifest_sha256'],'small')
    add('数据来源：https://grouplens.org/datasets/movielens/100k/；https://huggingface.co/datasets/mt0rm0/movie_descriptors。编码模型：meta-llama/Meta-Llama-3-8B，版本记录在历史提取元数据。代码：https://github.com/Jyonn/WSP4Rec。','small')
    path=ROOT/'output/pdf/WSP4Rec_description_complete_neutral_report.pdf';path.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(path),pagesize=(210*mm,297*mm),leftMargin=20*mm,rightMargin=20*mm,topMargin=20*mm,bottomMargin=23*mm,
        title='WSP4Rec 新数据集无方向提示实验报告').build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)


if __name__=='__main__':main()
