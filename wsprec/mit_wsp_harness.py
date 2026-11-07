"""Tool-free Codex turns and a non-executable numerical weight-edit language.

The model writes assignments, not runnable Python. Only this module's literal
AST interpreter can apply them. It cannot call functions or fit examples.
"""
from __future__ import annotations
import ast
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL='gpt-6-astra'
CLI='/Users/jyonn/.nvm/versions/node/v24.14.0/bin/codex'
TENSORS=('W1','b1','W2','b2','W3','b3')
FEATURES=('code_mode','collaboration_modes','shell_tool','unified_exec','unified_exec_tty',
          'shell_snapshot','view_image','image_generation','multi_agent','apps','plugins',
          'remote_plugin','browser_use','browser_use_external','browser_use_full_cdp_access',
          'in_app_browser','computer_use','code_mode_host','skill_search',
          'skill_mcp_dependency_install','sleep_tool','tool_suggest','goals','hooks',
          'worktrees','workspace_dependencies')


def zero_state(input_dim=22,width=32):
    return {'W1':np.zeros((input_dim,width),np.float32),'b1':np.zeros(width,np.float32),
            'W2':np.zeros((width,width),np.float32),'b2':np.zeros(width,np.float32),
            'W3':np.zeros((width,1),np.float32),'b3':np.zeros(1,np.float32)}


def forward(state,x):
    x=np.asarray(x,dtype=np.float32)
    h1=np.maximum(x@state['W1']+state['b1'],0)
    h2=np.maximum(h1@state['W2']+state['b2'],0)
    return (h2@state['W3']+state['b3']).reshape(-1)


def extract_edits(reply):
    blocks=re.findall(r'```([^\n`]*)\n(.*?)```',reply,flags=re.S)
    edits=[body for language,body in blocks if language.strip().lower()=='edits']
    if edits:
        if len(edits)!=1: raise ValueError('exactly one edits block required')
        return edits[0].strip()
    python=[body for language,body in blocks if language.strip().lower() in ('python','py')]
    if len(python)==1:return python[0].strip()
    if blocks:raise ValueError('no unambiguous edits block')
    # A plain assignment-only reply is accepted by the same strict interpreter.
    return reply.strip()


class _LiteralEditor:
    def __init__(self,state,max_abs):
        if set(state)!=set(TENSORS):raise ValueError('network requires six named tensors')
        self.state={k:np.array(v,dtype=np.float32,copy=True) for k,v in state.items()}
        self.max_abs=max_abs
    def integer(self,node):
        if isinstance(node,ast.Constant) and type(node.value) is int:return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=self.integer(node.operand);return -value if isinstance(node.op,ast.USub) else value
        raise ValueError('indices must be literal integers')
    def index(self,node):
        if isinstance(node,ast.Tuple):return tuple(self.index(v) for v in node.elts)
        if isinstance(node,ast.Slice):
            return slice(*(None if v is None else self.integer(v) for v in (node.lower,node.upper,node.step)))
        if isinstance(node,ast.Constant) and node.value is Ellipsis:return Ellipsis
        return self.integer(node)
    def reference(self,node):
        if isinstance(node,ast.Name) and node.id in self.state:return node.id,None
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id in self.state:
            return node.value.id,self.index(node.slice)
        raise ValueError('only existing tensor references are allowed')
    def expression(self,node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,(ast.List,ast.Tuple)):
            values=[self.expression(v) for v in node.elts]
            result=np.asarray(values,dtype=np.float64)
            if result.size>1000000:raise ValueError('literal array too large')
            return result
        if isinstance(node,(ast.Name,ast.Subscript)):
            name,index=self.reference(node)
            return self.state[name] if index is None else self.state[name][index]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=self.expression(node.operand);return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            left,right=self.expression(node.left),self.expression(node.right)
            with np.errstate(all='raise'):
                if isinstance(node.op,ast.Add):return left+right
                if isinstance(node.op,ast.Sub):return left-right
                if isinstance(node.op,ast.Mult):return left*right
                return left/right
        raise ValueError('only literals, tensor references and arithmetic are allowed')
    def apply(self,tree):
        if len(tree.body)>10000:raise ValueError('too many assignments')
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1:
                name,index=self.reference(node.targets[0]);value=self.expression(node.value)
            elif isinstance(node,ast.AugAssign) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                name,index=self.reference(node.target)
                value=self.expression(ast.BinOp(left=node.target,op=node.op,right=node.value))
            else:raise ValueError('only assignment and augmented assignment statements allowed')
            target=self.state[name] if index is None else self.state[name][index]
            value=np.asarray(value,dtype=np.float64)
            if not np.isfinite(value).all():raise ValueError('nonfinite assigned values')
            try:
                # Preserve every original parameter shape, including scalar views.
                value=np.broadcast_to(value,np.shape(target)).copy()
                if index is None:self.state[name][...]=value
                else:self.state[name][index]=value
            except (ValueError,IndexError,TypeError) as error:
                raise ValueError('assignment shape or index invalid') from error
            if not np.isfinite(self.state[name]).all() or np.max(np.abs(self.state[name]),initial=0)>self.max_abs:
                raise ValueError('nonfinite or oversized weight')
        return self.state


def apply_edits(state,reply,max_abs=1e6):
    code=extract_edits(reply)
    if len(code)>4*1024*1024:raise ValueError('edit text too large')
    try:
        tree=ast.parse(code,mode='exec')
        edited=_LiteralEditor(state,max_abs).apply(tree)
    except (SyntaxError,IndexError,TypeError,FloatingPointError,OverflowError,ZeroDivisionError) as error:
        raise ValueError('invalid literal edits: '+str(error)) from error
    delta=np.sqrt(sum(np.square(edited[k].astype(float)-np.asarray(state[k],dtype=float)).sum() for k in TENSORS))
    original=np.sqrt(sum(np.square(np.asarray(v,dtype=float)).sum() for v in state.values()))
    stats={'statements':len(tree.body),'changed_parameters':sum(int(np.count_nonzero(edited[k]!=state[k])) for k in TENSORS),
           'total_parameters':sum(v.size for v in edited.values()),'nonzero_parameters':sum(int(np.count_nonzero(v)) for v in edited.values()),
           'delta_l2':float(delta),'relative_delta_l2':float(delta/original) if original else None,
           'max_abs':max(float(np.max(np.abs(v),initial=0)) for v in edited.values())}
    return edited,stats


def audit_events(events):
    """Audit tools separately from recoverable transport/status notices.

    A reconnect/fallback notice is nonterminal only when a later completed turn
    and a saved agent-message event establish successful recovery. Explicit
    turn.failed, unknown error messages, and incomplete streams fail closed.
    """
    accepted={'agent_message','reasoning','error'}
    tools=[];usage=None;thread=None;unknown=[];terminal=[];notices=[]
    completed_index=-1;has_message=False
    def notice(message):
        return (bool(re.match(r'^Reconnecting\.\.\. \d+/\d+ \(',message))
                or message.startswith('Falling back from WebSockets to HTTPS transport.')
                or message.startswith('Code Mode is unavailable because code-mode host is disabled.'))
    for index,e in enumerate(events):
        kind=e.get('type','')
        if kind=='thread.started':thread=e.get('thread_id')
        if kind=='turn.completed':usage=e.get('usage');completed_index=index
        if kind=='turn.failed':terminal.append({'index':index,'type':kind,'message':e.get('error')})
        if kind=='error':
            entry={'index':index,'type':kind,'message':str(e.get('message',''))}
            (notices if notice(entry['message']) else terminal).append(entry)
        item=e.get('item')
        if isinstance(item,dict):
            item_type=item.get('type','')
            if item_type not in accepted:tools.append(item_type)
            if kind=='item.completed' and item_type=='agent_message':has_message=True
            if item_type=='error' and kind=='item.completed':
                entry={'index':index,'type':'item.error','message':str(item.get('message',''))}
                (notices if notice(entry['message']) else terminal).append(entry)
        if kind not in ('thread.started','turn.started','turn.completed','turn.failed','error','item.started','item.updated','item.completed'):
            unknown.append(kind)
    unresolved=[entry for entry in notices if entry['index']>completed_index]
    failed=bool(terminal or unresolved or completed_index<0 or not has_message)
    return {'tool_free':not tools and not unknown,'tool_item_types':tools,'unknown_event_types':unknown,
            'turn_failed':failed,'turn_completed':completed_index>=0,'agent_message_present':has_message,
            'terminal_errors':terminal,'nonterminal_notices':notices,'unresolved_notices':unresolved,
            'usage':usage,'thread_id':thread}


def cli_command(outdir,effort='xhigh',cli=CLI,resume_thread_id=None,ephemeral=True):
    outdir=Path(outdir).resolve();work=outdir/'empty_workdir';work.mkdir(parents=True,exist_ok=True)
    instruction=outdir/'instructions.txt';instruction.write_text('You are a helpful model.\n')
    cmd=[cli,'exec']
    if resume_thread_id:cmd+=['resume']
    cmd+=['--ignore-user-config','--skip-git-repo-check','--model',MODEL,'--json','-o',str(outdir/'answer.txt')]
    if ephemeral:cmd+=['--ephemeral']
    configs={'approval_policy':'never','sandbox_mode':'read-only','web_search':'disabled',
             'model_reasoning_effort':effort,'model_instructions_file':str(instruction),'agents.enabled':False}
    for feature in FEATURES:configs['features.'+feature]=False
    for key,value in configs.items():cmd+=['-c',key+'='+json.dumps(value)]
    if resume_thread_id:cmd+=[resume_thread_id,'-']
    else:cmd+=['-C',str(work),'-']
    return cmd,configs,work


def run_turn(prompt,outdir,effort='xhigh',timeout=3500,cli=CLI,resume_thread_id=None,ephemeral=True):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'run.json').exists():raise FileExistsError('turn already run')
    cmd,config,work=cli_command(outdir,effort,cli,resume_thread_id,ephemeral)
    (outdir/'prompt.txt').write_text(prompt)
    (outdir/'command.json').write_text(json.dumps({'argv':cmd,'cwd':str(work),'config':config},indent=2)+'\n')
    started=time.monotonic();timed_out=False
    try:version=subprocess.check_output([cli,'--version'],text=True,stderr=subprocess.DEVNULL,timeout=15).strip()
    except (OSError,subprocess.SubprocessError):version=None
    with (outdir/'events.jsonl').open('w') as stdout,(outdir/'stderr.txt').open('w') as stderr:
        p=subprocess.Popen(cmd,cwd=work,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,text=True)
        try:p.communicate(prompt,timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;p.kill();p.communicate()
    events=[];malformed=0
    for line in (outdir/'events.jsonl').read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:malformed+=1
    audit=audit_events(events);text=(outdir/'answer.txt').read_text() if (outdir/'answer.txt').exists() else ''
    record={'model_requested':MODEL,'effort_requested':effort,'cli_version':version,'returncode':p.returncode,'wall_seconds':time.monotonic()-started,
            'timed_out':timed_out,'audit':audit,'malformed_event_lines':malformed,
            'successful':p.returncode==0 and not timed_out and not malformed and audit['tool_free'] and not audit['turn_failed'] and bool(text),
            'configuration_warnings':[line for line in (outdir/'stderr.txt').read_text().splitlines() if 'unknown feature key' in line],
            'billing_usd':None,'scope':'Native Codex account; event token counters are not a monetary invoice.'}
    (outdir/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    return text,record


def run_trajectory(initial_prompt,initial_state,outdir,feedback:Callable|None=None,extra_forward_budget=0,effort='xhigh',cli=CLI,timeout=3500):
    """One candidate/turn, external practice-only forwards, final state retained.

    A zero-extra-forward repair may already include a baseline trace in its
    initial prompt. This is not called zero-feedback writing. Each turn is
    ephemeral; later turns replay the complete visible transcript rather than
    using MIT's persisted CLI resume protocol.
    """
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'trajectory.json').exists():raise FileExistsError('trajectory already exists')
    if type(extra_forward_budget) is not int or extra_forward_budget<0:raise ValueError('nonnegative integer budget required')
    started=time.monotonic();state={k:np.asarray(v,dtype=np.float32).copy() for k,v in initial_state.items()}
    np.savez(outdir/'initial_state.npz',**state);prompt=initial_prompt;turns=[];used=0;history=[{'role':'user','content':initial_prompt}]
    for turn in range(extra_forward_budget+1):
        remaining=max(1,timeout-(time.monotonic()-started))
        text,record=run_turn(prompt,outdir/f'turn{turn}',effort,remaining,cli,None,ephemeral=True)
        turns.append(record)
        if not record['successful']:
            result={'successful':False,'failure':'model_turn_failed','turns':turns,'forward_batches_used':used,'wall_seconds':time.monotonic()-started}
            break
        try:state,stats=apply_edits(state,text)
        except ValueError as error:
            result={'successful':False,'failure':'invalid_edits','error':str(error),'turns':turns,'forward_batches_used':used,'wall_seconds':time.monotonic()-started}
            break
        np.savez(outdir/f'state{turn}.npz',**state)
        (outdir/f'turn{turn}'/'edit_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
        if turn<extra_forward_budget:
            if feedback is None:raise ValueError('external feedback callback required')
            feedback_text=str(feedback(state));used+=1
            (outdir/f'feedback{used}.txt').write_text(feedback_text)
            history.extend([{'role':'assistant','content':text},{'role':'user','content':feedback_text}])
            prompt='Continue the same weight-editing task from this complete visible transcript. The latest user message is the current request. Prior edits are cumulative. Do not repeat already-applied edits unless you intend to apply them again. No tools are available.\n\n'+json.dumps(history,ensure_ascii=False,indent=2)
        else:
            np.savez(outdir/'final_state.npz',**state)
            result={'successful':True,'turns':turns,'forward_batches_used':used,'final_edit_stats':stats,'wall_seconds':time.monotonic()-started}
    result['conversation_mode']='ephemeral_stateless_visible_transcript_replay'
    if result['successful']:
        result['cumulative_edit_stats']={'changed_parameters':sum(int(np.count_nonzero(state[k]!=initial_state[k])) for k in TENSORS),
            'total_parameters':sum(v.size for v in state.values()),
            'delta_l2':float(np.sqrt(sum(np.square(state[k].astype(float)-np.asarray(initial_state[k],dtype=float)).sum() for k in TENSORS)))}
    (outdir/'trajectory.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
