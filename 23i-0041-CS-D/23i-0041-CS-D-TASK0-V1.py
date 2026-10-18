# Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
"""Reproduce all experiments. Standard Python library only.
Usage: python3 23i-0041-CS-D-TASK0-V1.py --build /tmp/pdc-build
       --graph /path/web-Google.txt --output /tmp/pdc-results
Build with the supplied Makefile first. Dataset must contain 5,105,039 edges.
Every C executable independently takes five measurements. Task 3 averages
50 kernels in each measurement. Results and heat fields stay outside the ZIP.
"""
import argparse, json, os, pathlib, subprocess, time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',required=True);p.add_argument('--graph',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OMP_PROC_BIND='true',OMP_PLACES='cores',OMP_WAIT_POLICY='PASSIVE',MPLCONFIGDIR='/tmp/pdc-mpl')
records=[]
def run(key,task,*args):
    cmd=[str(pathlib.Path(a.build)/f'task{task}'),*map(str,args)]
    print(time.strftime('%H:%M:%S'),key,flush=True)
    r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    (out/(key+'.txt')).write_text('$ '+' '.join(cmd)+'\n'+r.stdout)
    if r.returncode:raise RuntimeError(f'{key} exit {r.returncode}: {r.stdout}')
    records.append(dict(key=key,task=task,command=cmd,output=r.stdout))
    (out/'measurements.json').write_text(json.dumps(records,indent=2))
    return r.stdout
for t in (1,2,4):run(f't1_{t}',1,1<<24,t,12,'static')
for s in ('dynamic1','dynamic4096'):run('t1_'+s,1,1<<24,4,12,s)
run('t2_g6',2,'g6',4,'static')
for t in (1,2,4):
    for s in ('static','dynamic1024','guided'):run(f't2_{t}_{s}',2,a.graph,t,s)
run('t2_omit',2,a.graph,4,'static',1)
for t in (1,2,4):run(f't3_{t}',3,t,1024,*([a.graph] if t==4 else []))
run('t5_g6',5,'g6',4,0)
# Highest-ranked original page, obtained from THIS run of Task 2.
r=next(x for x in records if x['key']=='t2_4_static')['output']
source=int(next(line for line in r.splitlines() if line.startswith('TOP position=1 ')).split('id=')[1].split()[0])
run('t5_google',5,a.graph,4,source)
for n,s in ((512,20000),(2048,1000)):
    for v in ('scalar','simd'):run(f't4_{n}_{v}_1',4,n,s,1,v,0)
    for t in (1,2,4):
        for v in ('threads','hybrid','persistent'):
            field=[out/'heat512.txt'] if n==512 and v=='persistent' and t==4 else []
            run(f't4_{n}_{v}_{t}',4,n,s,t,v,1,*field)
    run(f't4_{n}_collapse_4',4,n,s,4,'collapse',1)
run('t4_2048_swapped_4',4,2048,1000,4,'swapped',1)
for init in (0,1):run(f't4_4096_init{init}',4,4096,200,4,'persistent',init)
# Check complete output agreement, not just the rounded assignment references.
import re, math
for n in (512,2048):
    baseline=next(x for x in records if x['key']==f't4_{n}_scalar_1')['output']
    expected=dict(re.findall(r'(centre|mean|delta)=([^\s]+)',baseline))
    for item in records:
        if item['task']==4 and item['key'].startswith(f't4_{n}_'):
            got=dict(re.findall(r'(centre|mean|delta)=([^\s]+)',item['output']))
            for field in expected:
                assert math.isclose(float(got[field]),float(expected[field]),rel_tol=1e-10,abs_tol=1e-12),(item['key'],field)
print('All required experiments and cross-variant checks completed.',flush=True)
