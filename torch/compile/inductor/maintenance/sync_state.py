"""仅由 state.json 渲染索引、每篇当前位置树和广度优先队列。"""
from pathlib import Path
import json,os,re
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'maintenance/state.json'
s=json.loads(path.read_text());nodes={n['id']:n for n in s['nodes']}
assert len(nodes)==len(s['nodes'])
def children(nid):return [n for n in s['nodes'] if n['parent']==nid]
queue=[];work=[s['root']]
while work:
    nid=work.pop(0);node=nodes[nid]
    if node['status']=='待展开':queue.append(nid)
    work.extend(n['id'] for n in children(nid))
s['breadth_first_queue']=queue
focus=s.get('priority_focus',{}).get('nodes',[])
assert len(focus)==len(set(focus)) and all(nid in nodes for nid in focus)
def focus_rank(nid):
    return next((i for i,root in enumerate(focus) if nid==root or nid.startswith(root+'/')),None)
priority=sorted((nid for nid in queue if focus_rank(nid) is not None),key=lambda nid:(nid.count('/'),focus_rank(nid),queue.index(nid)))
reading=priority+[nid for nid in queue if nid not in priority]
s['reading_queue']=reading;s['next']=reading[0] if reading else None
path.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
lines=['# 当前阅读索引','', '模块身份、状态与顺序来自 [state.json](maintenance/state.json)。本层完成表示已讲清直接子模块交接；只有“叶子·逐行完成”才表示相应完整源码范围已逐行解释。','']
def render(nid,depth=0):
    n=nodes[nid];label=n['title']
    if n.get('path'):label=f'[{label}]({n["path"]})'
    lines.append('  '*depth+f'- {label}：{n["status"]}')
    for c in children(nid):render(c['id'],depth+1)
render(s['root'])
if focus:
    lines+=['','按用户要求，优先展开支持新设备相关模块：后端注册与调度、图重写的模式注册/匹配/子图替换（自定义融合算子）、wrapper 设备调用、Lowering 设备状态、后端能力及编译加载，以及布局/物化/alias/mutation、依赖/副作用/生命周期与异步释放。已完成当前图收尾；设备相关子树优先，其余范围保留。','']
lines+=['','当前待写顺序（设备接入优先，其余保持广度优先）：' if focus else '广度优先待写顺序：','']
for i,nid in enumerate(reading,1):
    n=nodes[nid];tag='〔设备接入优先〕' if nid in priority else '';lines.append(f'{i}. {n["title"]}（{nodes[n["parent"]]["title"]}）{tag}')
(ROOT/'INDEX.md').write_text('\n'.join(lines)+'\n')
for nid,n in nodes.items():
    if not n.get('path'):continue
    f=ROOT/n['path'];assert f.is_file(),f
    chain=[];cur=n
    while cur is not None:
        chain.append(cur);cur=nodes.get(cur['parent'])
    chain.reverse();tree=[]
    for depth,ancestor in enumerate(chain):
        label=ancestor['title']+(' ← 当前篇' if ancestor['id']==nid else '')
        tree.append('    '*depth+('└── ' if depth else '')+label+'  '+ancestor['status'])
    for i,c in enumerate(children(nid)):
        tree.append('    '*len(chain)+('└── ' if i==len(children(nid))-1 else '├── ')+c['title']+'  '+c['status'])
    content=re.sub(r'\n\n<!-- child-links:start -->.*?<!-- child-links:end -->','',f.read_text(),flags=re.S)
    content=re.sub(r'```text\n.*?\n```','```text\n'+'\n'.join(tree)+'\n```',content,count=1,flags=re.S)
    linked=[c for c in children(nid) if c.get('path')]
    links='直接子篇：'+'、'.join(f'[{c["title"]}]({os.path.relpath(ROOT/c["path"],f.parent)})' for c in linked)+'。' if linked else ''
    block='<!-- child-links:start -->\n'+links+'\n<!-- child-links:end -->'
    tree_match=re.search(r'```text\n.*?\n```',content,flags=re.S)
    assert tree_match is not None,f
    end=tree_match.end();content=content[:end]+'\n\n'+block+content[end:]
    f.write_text(content)
