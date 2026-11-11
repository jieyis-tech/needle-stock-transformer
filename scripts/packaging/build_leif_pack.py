"""Explicit-allowlist teammate handoff. Creates files only; never sends them."""
from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'deliverables'
base='Leif_Checkin_Pack_2026-11-10'
name=base;version=1
while (OUT/name).exists() or (OUT/(name+'.zip')).exists():
    version+=1;name=base+f'_v{version}'
PACK=OUT/name;PACK.mkdir(parents=True)


def write(rel,s):
    p=PACK/rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(s.strip()+'\n',encoding='utf-8')


def copy(src,dest):
    p=PACK/dest;p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/src,p)


for p in (ROOT/'checkin/leif_handoff_notes').glob('*.md'):
    copy(p.relative_to(ROOT).as_posix(),p.name)
copy('checkin/slides/Needle_Transformer_Checkin.pptx','slides/Needle_Transformer_Checkin.pptx')
copy('checkin/Speaker_Notes.md','slides/Speaker_Notes_English_Chinese.md')
for i in (1,2,3):copy(f'checkin/.build/final-slide-{i}.png',f'slides/slide-{i}.png')

notes=(ROOT/'checkin/Speaker_Notes.md').read_text(encoding='utf-8')
sections=[('Slide 1',notes.split('## 第 1 页：',1)[1].split('## 第 2 页：',1)[0]),
          ('Slide 2: implementation',notes.split('### 先介绍实现',1)[1].split('### 切换到演示',1)[0]),
          ('Slide 2: demo',notes.split('### 看到四条 PASS 后',1)[1].split('**中文对照：**',1)[0]),
          ('Slide 3',notes.split('## 第 3 页：',1)[1].split('## 导师可能追问',1)[0])]
english=['# Presenter script in English','The presenter uses this script. Leif can use his speaking card for questions.']
for title,s in sections:
    paras=[p.strip() for p in s.split('\n\n') if re.match(r'^[A-Za-z]',p.strip())]
    english+=['## '+title]+paras
write('slides/Presenter_Script_English.md','\n\n'.join(english))

# Readable source, no source databases, binary runtimes or original local launcher.
for folder in ('master_needle','vendor/needle'):
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.cu') and '__pycache__' not in p.parts:
            rel=p.relative_to(ROOT).as_posix();copy(rel,'code/'+rel)
provenance=json.loads((ROOT/'vendor/PROVENANCE.json').read_text(encoding='utf-8'))
provenance['source']='Completed local HW2 Needle snapshot; original workstation path omitted from teammate pack.'
write('code/vendor/PROVENANCE.json',json.dumps(provenance,indent=2))
for rel in ('checkin/demo_checkin.py','scripts/benchmark_gpu.py','scripts/verify_gpu.py','scripts/check_training_numerics.py','tests/test_master.py'):
    copy(rel,'code/'+rel)
write('code/requirements-demo.txt','numpy>=1.26\ntorch>=2.0\nthreadpoolctl>=3.1')
write('code/CODE_MAP.md','''# Code map

Read model.py for attention axes; attention.py for TensorOp dispatch;
cuda_attention.py for launches; kernels/attention.cu for forward/dQ/dK-dV;
backend.py for device binding/first-order gradients. All are in master_needle/.
vendor/needle is the unchanged course-HW2 dependency, not new project work.

Optional command from this folder: python checkin/demo_checkin.py --backend cpu
No stock panel or CUDA binaries are included. See ../04_Demo_Guide.md.
Other GPU/data scripts are code-reading references and need the omitted runtime/panel.
''')
launcher_lines=['@echo off','setlocal','chcp 65001 >nul','title Needle CPU Demo - Leif Preparation',
               'pushd "%~dp0"','where py >nul 2>nul','if not errorlevel 1 (',
               '  py -3 -X utf8 checkin\\demo_checkin.py --backend cpu',') else (',
               '  python -X utf8 checkin\\demo_checkin.py --backend cpu',')',
               'set "NM_RESULT=%ERRORLEVEL%"','popd','echo.','pause','exit /b %NM_RESULT%']
(PACK/'code/Run_CPU_Demo.cmd').write_bytes(('\r\n'.join(launcher_lines)+'\r\n').encode('ascii'))

bench=json.loads((ROOT/'results/gpu_benchmark.json').read_text(encoding='utf-8'))
write('evidence/gpu_benchmark.json',json.dumps(dict(device=bench['device'],measured_at_utc=bench.get('measured_at_utc'),rows=bench['rows'],scope='Original presenter experiment; live CuPy allocations, not whole-device memory.'),indent=2))
for file in ('gpu_correctness.json','training_numerics.json','real_alignment.json'):
    copy('results/'+file,'evidence/'+file)
logs=[]
for p in (ROOT/'checkin/demo_runs').glob('*.json'):
    r=json.loads(p.read_text(encoding='utf-8'))
    if r.get('backend')=='cuda' and r.get('status')=='passed':logs.append((p.name,r))
assert logs,'No saved passing CUDA demo found'
write('evidence/presenter_demo_result.json',json.dumps(sorted(logs)[-1][1],indent=2))
write('evidence/Results_At_A_Glance.md','''# Results at a glance

These are saved presenter results, not runs on Leif's machine.

| Case | Result | Interpretation |
|---|---|---|
| Tiny CUDA attention | Output error ~2.38e-7, Q/K/V error ~4.17e-7 | One small reference case |
| Model forward | 32 stocks x 12 days -> 32 finite scores | Functional check |
| Synthetic 40-update fit | Loss ~1.1209 -> 0.0077 | Same-batch learning check |
| Full model: dense handwritten VJP | 397.0 MiB / 15.66 ms | GPU benchmark baseline |
| Full model: tiled recomputation | 158.7 MiB / 21.81 ms | Lower memory, longer step |

Full-model benchmark: 1024 stocks, T=12, FP32, width=32, block=64;
RTX 5080 Laptop GPU; 2 warmups and 5 timed iterations on 2026-10-26.
Memory is live CuPy-pool allocation, excluding context/external workspaces.
The separate real training experiment has 128 historical stocks, one seed,
three fixed epochs. We do not claim improved prediction or trading returns.

FP32 long trajectories diverge. A train-only 16-step check found parameter
differences around 1.27e-3 in FP32 and around 1e-12 in FP64. Same-kernel replay
was exact for that test, but this does not prove every long-run difference is explained.
''')
copy('docs/IMPLEMENTATION.md','background/Implementation_Notes.md')
copy('docs/REPORT_DRAFT.md','background/Report_Draft.md')
write('meeting/Meeting_Checklist.md','''# Mentor meeting checklist

- Whole group attends; reschedule with the mentor if someone cannot attend.
- Prepare a 2-3 minute pitch: idea, Needle implementation, progress.
- Have slides and the short demo ready; rehearse once before the meeting.
- Know your responsibility and the overall structure; name background sources.
- Remaining work: review numerical/correctness results, summarize evaluation, notebook/video.
- Check-in: November 14. The saved notice separately sets November 16 at 11:59 PM
  for the TA feedback PDF upload. Follow scheduling emails and later announcements.
- One member uploads the TA-provided PDF to Gradescope and adds both group members;
  do not make a duplicate submission from the other member.
- Video: 90 seconds, November 28. Report: Colab notebook, about 2000 words plus
  private code access, December 7. Colab execution is not yet verified.
- The TA form is feedback from the TA, not self-issued approval.

This pack does not mean mentor approval or final submission. Keep course code private.
''')
form=ROOT.parent/'Mentor Check-in/dlsys_checkin_form.pdf'
if form.is_file():
    p=PACK/'meeting/TA_Checkin_Form.pdf';shutil.copy2(form,p)


def render(md):
    result=[];para=[]
    def safe(s):return re.sub(r'`([^`]+)`',r'<code>\1</code>',html.escape(s))
    def flush():
        if para:result.append('<p>'+safe(' '.join(para))+'</p>');para.clear()
    for line in md.splitlines():
        if not line.strip():flush()
        elif line.startswith('#'):
            flush();level=min(len(line)-len(line.lstrip('#'))+1,4)
            result.append(f'<h{level}>'+safe(line.lstrip('#').strip())+f'</h{level}>')
        elif line.startswith('- '):flush();result.append('<p class="item">• '+safe(line[2:])+'</p>')
        else:para.append(line.strip())
    flush();return '\n'.join(result)


header='''<h1>Needle Transformer: Leif's check-in pack</h1>
<p class="lead">Understand the project, your GPU-focused role, and likely mentor questions.</p>
<p>Prepared 10 November 2026. Reading needs no Python or CUDA installation. Private teammate handoff.</p>
<h2>Read this first</h2><ol><li>Read the project overview and your speaking card.</li>
<li>Skim the likely questions. Explain them in your own words.</li>
<li>View the three slides, then follow the GPU walkthrough if needed.</li></ol>
<p><a href="#overview">Overview</a> · <a href="#card">Speaking card</a> · <a href="#qa">Questions</a> · <a href="#slides">Slides</a></p>
<p><strong>397 → 159 MiB</strong>: lower live GPU-array allocation in one test.<br>
<strong>16 → 22 ms</strong>: more time per step. This is a memory/time trade-off.</p>
<p><a href="slides/Needle_Transformer_Checkin.pptx">Open PowerPoint</a> ·
<a href="slides/Presenter_Script_English.md">Presenter's English script</a> ·
<a href="slides/Speaker_Notes_English_Chinese.md">Full current speaker notes</a></p>'''
body=header
for ident,file in [('overview','01_Project_Overview.md'),('card','03_Leif_Speaking_Card.md'),('qa','02_Leif_Checkin_QA.md'),('gpu','05_GPU_Walkthrough.md'),('demo','04_Demo_Guide.md')]:
    body+=f'<section id="{ident}">'+render((PACK/file).read_text(encoding='utf-8'))+'</section>'
body+='<section id="slides"><h2>Slide previews</h2>'+''.join(f'<img src="slides/slide-{i}.png" alt="Slide {i}">' for i in (1,2,3))+'</section>'
body+='''<h2>References</h2><p><a href="https://doi.org/10.1609/aaai.v38i1.27767">MASTER, Li et al., AAAI 2024</a><br>
<a href="https://arxiv.org/abs/2205.14135">FlashAttention, Dao et al., 2022</a><br>
Existing ideas are attributed; the project contribution is Needle implementation and evaluation.</p>'''
write('START_HERE.html','''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Leif — Needle check-in</title><style>
body{font:17px/1.6 system-ui,Arial,sans-serif;max-width:960px;margin:42px auto;padding:0 25px;color:#182c43}h1{font-size:34px;line-height:1.2}h2{font-size:25px;margin-top:38px;border-bottom:1px solid #dce4e9;padding-bottom:7px}h3{font-size:20px;margin-top:26px}a{color:#087f8c}.lead{font-size:21px;color:#51616e}.item{margin:5px 0}img{display:block;max-width:100%;height:auto;margin:25px 0;border:1px solid #e1e6ea}li{margin:6px 0}p{margin:12px 0}code{font-size:90%}
</style><body>'''+body+'</body></html>')
write('README.txt','''LEIF CHECK-IN PACK

Extract the whole ZIP, then double-click START_HERE.html. It works offline.
The main guide and Q&A are in English. Slides and source files are included.
No setup is required for reading. The optional CPU demo needs Python dependencies;
see 04_Demo_Guide.md. There is no stock panel, CUDA runtime, credential file or
original machine-specific Python launcher in this pack.

Keep it private within the project group because it includes Needle course code.
''')

files=[p for p in PACK.rglob('*') if p.is_file()]
for p in files:
    rel=p.relative_to(PACK)
    assert not any(s in rel.parts for s in ('.deps','.git','__pycache__','demo_runs','.verification')),rel
    assert rel.parts[0]!='data' and rel.parts[:2]!=('code','data'),rel
    assert p.suffix.lower() not in ('.dll','.so','.pyd','.env','.npz','.duckdb','.pt','.pyc'),rel
manifest=dict(prepared_date='2026-11-10',purpose='Private teammate orientation and check-in preparation',
              source_snapshot='Current local project; benchmark 2026-10-26; demo evidence 2026-11-06',
              files={p.relative_to(PACK).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
write('MANIFEST.json',json.dumps(manifest,indent=2))
archive=OUT/(name+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PACK.rglob('*')):
        if p.is_file():z.write(p,name+'/'+p.relative_to(PACK).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(manifest['files'])+1
print(json.dumps(dict(folder=str(PACK),zip=str(archive),zip_bytes=archive.stat().st_size,
                      files=len(manifest['files'])+1,zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2,ensure_ascii=False))
