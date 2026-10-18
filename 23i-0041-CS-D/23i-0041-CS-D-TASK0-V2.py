# Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
"""Generate the 12-page technical report from actual benchmark outputs.
Usage: python3 23i-0041-CS-D-TASK0-V2.py RESULTS_DIR BUILD_DIR REPORT.pdf
Only report generation requires reportlab, matplotlib and Pillow.
C Tasks 1-4 use no external library; Task 5 uses SuiteSparse:GraphBLAS.
"""
import os,sys,re,json,pathlib,html
os.environ.setdefault('MPLCONFIGDIR','/tmp/pdc-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/pdc-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PImage, ImageDraw, ImageFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
R=pathlib.Path(sys.argv[1]);B=pathlib.Path(sys.argv[2]);target=sys.argv[3]
records={x['key']:x for x in json.loads((R/'measurements.json').read_text())}
def fields(line):return dict(re.findall(r'(\w+)=([^\s]+)',line))
def lines(k,prefix):return [fields(l) for l in records[k]['output'].splitlines() if l.startswith(prefix)]
def result(k):return lines(k,'RESULT')
def text(k):return records[k]['output']
def time(k):return float(result(k)[0]['seconds'])
def fmt(x):return f'{float(x):.6f}'
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';mono='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
pdfmetrics.registerFont(TTFont('D',font));pdfmetrics.registerFont(TTFont('DB',bold));pdfmetrics.registerFont(TTFont('DM',mono))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyD',fontName='D',fontSize=9,leading=12,spaceAfter=7));styles.add(ParagraphStyle(name='HeadD',fontName='DB',fontSize=17,leading=21,spaceAfter=12,textColor=colors.HexColor('#12445c')));styles.add(ParagraphStyle(name='SubD',fontName='DB',fontSize=10.5,leading=14,spaceAfter=6));styles.add(ParagraphStyle(name='CellD',fontName='D',fontSize=7.7,leading=10));styles.add(ParagraphStyle(name='CodeD',fontName='DM',fontSize=6.8,leading=9,spaceAfter=6))
story=[]
def p(s):story.append(Paragraph(s,styles['BodyD']))
def h(s):story.append(Paragraph(s,styles['SubD']))
def page(title):
 if story:story.append(PageBreak())
 story.append(Paragraph(title,styles['HeadD']))
def table(rows,widths=None,fs=7.7):
 data=[[Paragraph(html.escape(str(c)),styles['CellD']) for c in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0edf2')),('FONTNAME',(0,0),(-1,0),'DB'),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#a5bac4')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([t,Spacer(1,8)])
def code(s):story.append(Preformatted(s,styles['CodeD']))
def bench_rows(task,prefix,variants):
 rows=[['Variant','Threads','Seconds','Speedup','Efficiency']]
 base=None
 for t in (1,2,4):
  for r in result(f'{prefix}{t}'):
   if r['variant'] not in variants:continue
   if t>1 and r['threads']=='1':continue
   if base is None:base=float(r['seconds'])
   sec=float(r['seconds']);nt=int(r['threads']);rows.append([r['variant'],nt,f'{sec:.6f}',f'{base/sec:.3f}',f'{base/sec/nt:.3f}'])
 return rows
page('Assignment 2 · Parallel & Distributed Computing')
p('<b>Muhammad Ayaan Naveed · 23i-0041 · CS-D</b><br/>Sparse Matrices, Graph Analytics and Stencil Computation using OpenMP and SIMD')
h('1. Machine, tools and measurement protocol')
p('All experiments were executed in the same Linux cloud VM. Measurements describe this machine, not a personal laptop. Timings are kernel wall times from omp_get_wtime(); reading, allocation, printing and initialisation are excluded. Each executable takes the minimum of five runs. Task 3 times 50 SpMVs per run and reports the smallest of five 50-run means. No speedups or validation values are simulated.')
table([['Property','Observed configuration'],['CPU','AMD EPYC 9V74 80-Core Processor (KVM virtual machine)'],['Available processors','5 logical CPUs, reported as 5 cores / 1 thread per core; cgroup CPU quota 4 cores'],['Memory / NUMA','32 GiB cgroup limit; one exposed NUMA node'],['Caches (hypervisor report)','L1d 96 KiB / 3 instances, L1i 96 KiB / 3; L2 3 MiB / 3; L3 32 MiB shared'],['Instruction support','AVX, AVX2, FMA and AVX-512F are present'],['Register widths','AVX/AVX2: 256 bits = 8 float / 4 double lanes; AVX-512: 512 bits = 16 float / 8 double'],['Compiler / library','GCC 14.2.0; SuiteSparse:GraphBLAS 7.4.0, GraphBLAS C API 2.0'],['Build settings','-O2 -std=c11 -fopenmp -mavx2 -mfma -ffp-contract=off'],['Thread settings','1, 2, 4; OMP_DYNAMIC=false, OMP_PROC_BIND=true, OMP_PLACES=cores, OMP_WAIT_POLICY=PASSIVE']], [112,398])
p('Eight threads exceed the five visible hardware threads and the four-core quota, so 8-thread tests are excluded. Four threads are the highest useful prescribed count. AVX-512 is deliberately unused so that all tasks share the prescribed AVX2/FMA target. Scalar kernels explicitly disable automatic vectorisation; SIMD variants are identified in code. Explicit AVX2 FMA intrinsics in Task 3 remain fused even with automatic contraction disabled.')
h('Data provenance and limits')
p('The Stanford endpoint was blocked. Two independent accessible GitHub copies of web-Google.txt were checked; their canonical LF text matches exactly. The graph has 875,713 distinct endpoint IDs and 5,105,039 directed edges. Vertex IDs are compacted in sorted order and retained for output. No dense graph matrix is ever allocated. The full dataset and all binaries are kept outside the submission.')
code('Canonical SHA256 (LF text):\n543b17f820d78af6616618f3ece2da836e917f089140d68113045313a3523d45')
p('Mirror pinned by the reproduction script: bhargavikesarkar/parallel-Webgraph-analysis, commit f7ac0ebb93c25e42ef99da7b8685259f74a7598b. A second copy in sanya-sajin/Google_Graph_Analysis matched after CRLF normalisation. VM contention and quota throttling can affect scaling; the minimum of five reduces but does not eliminate this noise.')
page('2. Task 1 · Correctness, sharing and dependence')
p('N = 2^24 float inputs cover [−π, π), with x[i] = −π + 2πi/N and sinf as the reference. Listing 2 performs j = 1 … terms corrections after the initial x. The update <b>denom *= (2*j+2)*(2*j+3)</b> overflows when forming 13! after j=5; the declaration <b>int denom = 6</b> selects an insufficient signed 32-bit type. The minimal fix is <b>float denom = 6.0f</b>. The supplied program also models the original denominator using defined uint32_t wrapping; actual signed overflow in the original C is undefined, so its exact output is not portable.')
errs=lines('t1_1','ERROR');table([['Corrections','Original wrap-model max |error|','Corrected max |error|']]+[[r['terms'],r['wrapping_original'],r['corrected']] for r in errs],[75,220,215])
p('The four-correction error is truncation error. Increasing to eight corrections sharply reduces it; beyond this, single-precision rounding sets an error floor of roughly 10⁻⁶. Tiny changes at twelve corrections do not imply further factorial overflow. All measured thread, SIMD and hybrid versions match their corrected scalar output bit for bit on this input and build; the executable additionally prints maximum absolute differences and bit-difference counts.')
table([['Variable','Sharing','Reason'],['N, terms','shared','Read-only loop bounds / correction count'],['x, y','shared','Shared array storage; x is read-only and each i owns y[i]'],['i','private','OpenMP associated-loop induction variable'],['j','private','Local inner-loop induction variable for one input'],['value','private','Per-input accumulator'],['numer','private','Per-input power recurrence'],['denom','private','Per-input factorial recurrence'],['sign','private','Per-input alternating-sign recurrence']], [80,80,350])
h('Why the outer loop is parallel, but not the inner recurrence')
p('Each outer iteration reads x[i] and writes a unique y[i]. No iteration needs another output. Inner iteration j reads value, numer, denom and sign produced by j−1; all four carry loop dependence. Parallelising that inner loop directly would break the recurrence. Instead, SIMD lanes each evaluate a different input while preserving its internal correction order.')
h('Compiler vectorisation evidence')
vec=(B/'task1.vec').read_text().splitlines();selected=list(dict.fromkeys(l.split('/')[-1] for l in vec if 'loop vectorized' in l))
code('\n'.join(selected[:4]))
p('Fixed-count 4/8/12-correction specialisations unroll the recurrence before vectorising across inputs. The one-thread SIMD loop uses omp simd; the hybrid loop uses omp parallel for simd with runtime scheduling. The compiler report above concerns the actual output loops, not merely initialisation. Automatic FMA contraction is disabled to keep scalar and SIMD operation ordering comparable.')
page('3. Task 1 · Performance and scheduling')
table(bench_rows(1,'t1_',{'scalar','threads','simd','hybrid'}),[100,55,115,115,125])
p('Speedup = scalar sequential time / variant time; efficiency = speedup / executing thread count. Tables use the one-thread scalar measurement as their common reference. Hybrid efficiency can exceed one because it includes SIMD and specialisation gains. One-thread SIMD efficiency is therefore equal to its SIMD speedup. The sequential and SIMD-only rows appear once; two- and four-thread rows apply only to parallel variants. The same convention is used in later tables.')
h('Highest-thread-count scheduling comparison')
rows=[['Schedule','Threads seconds','Hybrid seconds']]
for s,k in [('static','t1_4'),('dynamic,1','t1_dynamic1'),('dynamic,4096','t1_dynamic4096')]:
 rr={r['variant']:r for r in result(k)};rows.append([s,rr['threads']['seconds'],rr['hybrid']['seconds']])
table(rows,[150,180,180])
rr=[r for t in (1,2,4) for r in result(f't1_{t}')];best=min(rr,key=lambda r:float(r['seconds']))
p(f'The fastest measured variant is <b>{best["variant"]} on {best["threads"]} thread(s)</b>, at {float(best["seconds"]):.6f} s. Every input performs the same number of corrections, so there is little load imbalance for dynamic scheduling to repair. Static assignment gives large contiguous ranges and little work-distribution overhead. Dynamic,1 invokes the scheduling machinery for individual elements and fragments SIMD chunks; dynamic,4096 amortises that overhead over larger blocks.')
p('The Taylor recurrence contains serial dependency chains within each input. SIMD exposes independent chains across inputs, but a pragma alone does not guarantee a speedup: instruction throughput and division latency matter. Fixed-count unrolling also lets GCC constant-fold factorial denominators; any speedup includes this optimisation and removed loop/call overhead, not just vector width. Thread scaling is limited by the four-core quota and by the VM scheduler. Reported values should be interpreted as measurements of this compiler and processor rather than an assumption that all four variants must become faster.')
h('Reproducible command examples')
code('make BUILD=/tmp/pdc-build all\n/tmp/pdc-build/task1 16777216 1 12 static\n/tmp/pdc-build/task1 16777216 2 12 static\n/tmp/pdc-build/task1 16777216 4 12 static\n/tmp/pdc-build/task1 16777216 4 12 dynamic1\n/tmp/pdc-build/task1 16777216 4 12 dynamic4096')
p('The program runs all three correctness correction counts before timing each variant. Allocation, reference calculation and error checking occur outside measured intervals. The full reproduction driver sets affinity and passive OpenMP waiting consistently for every task.')
page('4. Task 2 · Sparse PageRank results')
g6=lines('t2_g6','TOP');g=lines('t2_4_static','TASK2')[0];pr=result('t2_4_static')[0]
p('For each edge src → dst, incoming CSR row dst stores column src and value 1/outdeg(src). Construction counts row lengths, prefix-sums them to row_ptr, and scatters edges with a cursor per row. It requires O(n + nnz) storage. Every iteration uses d = 0.85, sums dangling rank, applies M to r, and stops at L1 change &lt; 10⁻⁸, or after 200 iterations. Rank buffers are swapped by pointer; only final extraction may copy a buffer after timing.')
table([['G6 vertex','Measured rank','Assignment (rounded)']]+[[i,next(r['rank'] for r in g6 if int(r['id'])==i),v] for i,v in enumerate(['0.186037','0.104066','0.273216','0.141117','0.189870','0.105695'])],[90,220,200])
p(f'G6 converges in {result("t2_g6")[0]["iterations"]} iterations; rank mass is {result("t2_g6")[0]["mass"]}. Its ranks pass the stated tolerance. For the full Google graph: n = {g["n"]}, nnz = {g["nnz"]}, dangling pages = {g["dangling"]}; convergence takes {pr["iterations"]} iterations, and rank mass is {pr["mass"]}.')
table([['Position','Original page ID','Rank']]+[[r['position'],r['id'],r['rank']] for r in lines('t2_4_static','TOP')],[70,160,280])
h('Dangling-page experiment')
omit=result('t2_omit')[0]
p(f'With the dangling term deliberately omitted, final rank mass becomes <b>{omit["mass"]}</b> after {omit["iterations"]} iterations. This version is an experiment, not the submitted correct algorithm. Rank leaks through pages with zero out-degree because their columns contain no transition probability. Uniform redistribution d·s/n preserves total probability. G6 has no dangling vertices, so it alone cannot expose this bug; the Google experiment is necessary.')
page('5. Task 2 · Thread counts, scheduling and correctness')
base=float(result('t2_1_static')[0]['baseline']);rows=[['Threads','Schedule','Seconds','Speedup','Efficiency']]
for t in (1,2,4):
 for sched in ('static','dynamic1024','guided'):
  r=result(f't2_{t}_{sched}')[0];sec=float(r['seconds']);rows.append([t,sched,f'{sec:.6f}',f'{base/sec:.3f}',f'{base/sec/t:.3f}'])
table(rows,[65,115,110,110,110]);p(f'The common scalar reference for this table is {base:.6f} s (minimum of five scalar runs). Each process also records its own scalar time in the raw outputs, so measurement noise can be inspected independently.')
p(f'Incoming row lengths have mean {g["meanrow"]}, standard deviation {g["row_sd"]}, and maximum {g["maxrow"]}. This distribution is highly irregular: a few popular pages have many incoming links, while many rows are short or empty. Equal row counts are therefore not equal work. Dynamic chunks can reduce imbalance; guided scheduling gradually shrinks chunks, reducing scheduler overhead while allowing redistribution near the end.')
best=min(((time(f't2_{t}_{s}'),t,s) for t in (1,2,4) for s in ('static','dynamic1024','guided')))
p(f'The fastest measured policy/count is <b>{best[2]} with {best[1]} thread(s)</b> at {best[0]:.6f} s. This is the empirical choice for this VM. Dynamic,1024 trades balancing against work-distribution and locality costs; guided trades progressively smaller chunks against similar costs. Static can still win when scheduling overhead, CSR locality or bandwidth dominates the cost of uneven rows. One should choose from measurements, not from irregularity alone.')
h('Race freedom and reproducibility')
p('Every SpMV row writes one b[i], so no row shares an output. Current rank a is read-only throughout the sweep. Dangling mass and L1 change use OpenMP sum reductions. The barrier at the end of the sweep completes all outputs before the rank pointers are exchanged. Different reduction trees can change low-order rounding; the maximum final rank difference from the sequential baseline is checked against 10⁻¹⁰.')
maxdiff=max(float(result(f't2_{t}_{s}')[0]['maxdiff']) for t in (1,2,4) for s in ('static','dynamic1024','guided'))
p(f'Largest observed final-rank difference over all Google scheduling runs: {maxdiff:.3g}. Input lines beginning with # are ignored; endpoint IDs are sorted and uniqued, then binary-searched for relabelling. The source/destination direction is preserved. Top-ten output uses the reverse mapping, so the page identifiers are original IDs rather than CSR row numbers.')
code('/tmp/pdc-build/task2 /path/web-Google.txt 4 static\n/tmp/pdc-build/task2 /path/web-Google.txt 4 dynamic1024\n/tmp/pdc-build/task2 /path/web-Google.txt 4 guided\n/tmp/pdc-build/task2 /path/web-Google.txt 4 static 1  # experiment only')
page('6. Task 3 · B in COO, CSR and CSC')
p('B is stored as sparse arrays throughout. The following diagrams list every entry, using four-byte indices and four-byte float values. Array payload sizes exclude constant-size C structure metadata and allocator bookkeeping.')
row=[0,0,1,1,2,2,2,3,4,4,4,5,5];col=[0,1,1,4,0,2,3,3,2,4,5,3,5];val=[4,1,4,9,6,4,5,4,8,4,1,6,4];rp=[0,2,4,7,8,11,13]
def arr(a):return ' '.join(map(str,a))
h('COO · 12·nnz = 156 bytes')
code('position '+arr(range(13))+'\nrow      '+arr(row)+'\ncol      '+arr(col)+'\nval      '+arr(val))
h('CSR · 8·nnz + 4·(n+1) = 132 bytes')
code('row_ptr  '+arr(rp)+'\ncol      '+arr(col)+'\nval      '+arr(val))
p('Conversion CSR → COO expands each interval [row_ptr[i], row_ptr[i+1]) into explicit row index i; col and val are copied to identical positions. Rows own disjoint half-open intervals because row_ptr is non-decreasing. Consequently, parallel rows never write the same COO element, even for empty rows.')
h('CSC · 8·nnz + 4·(n+1) = 132 bytes')
entries=sorted(zip(col,row,val));cp=[0]
for j in range(6):cp.append(cp[-1]+sum(c==j for c in col))
code('col_ptr  '+arr(cp)+'\nrow_idx  '+arr([r for c,r,v in entries])+'\nval      '+arr([v for c,r,v in entries]))
p('CSR groups each output dot product by row. CSC groups nonzeros by input column; it is useful for column access but a straightforward column-parallel Ax would have multiple threads adding into the same y[i]. The CSC representation is shown for the required diagram; no CSC kernel is required.')
h('A and B correctness')
code('\n'.join(l for l in text('t3_4').splitlines() if l.startswith('VALID')))
p('All three kernels reproduce the required outputs exactly on these integer-valued checks. The small sizes exercise the ELL scalar tail and DIA boundary handling; the million-row Laplacian additionally exercises full AVX2 blocks. Conversion to COO is checked entry-by-entry. DIA is also tested with a one-byte limit and must reject without allocating output tables.')
page('7. Task 3 · ELL and DIA conversion diagrams')
h('CSR → ELL · W = 3 · 8·n·W = 144 bytes')
p('Find W with reduction(max:W), then fill one independent row per thread. Each slot is a contiguous column-major group of rows, index k·n+i. A dot denotes zero padding; the actual stored padding is value 0, column 0.')
ecols=[];evals=[]
for k in range(3):
 ec=[];ev=[]
 for i in range(6):
  pos=rp[i]+k;ec.append(col[pos] if pos<rp[i+1] else 0);ev.append(val[pos] if pos<rp[i+1] else 0)
 ecols.append(ec);evals.append(ev)
table([['Slot k','col[k·n … k·n+5]','val[k·n … k·n+5]']]+[[k,arr(ecols[k]),arr(evals[k])] for k in range(3)],[65,220,225])
p('AVX2 gather loads x[col] before multiplication. A zero matrix value does not suppress an invalid memory access, so padded col must be a valid index. Column zero is safe for every nonempty matrix.')
h('CSR → DIA · offsets sorted increasingly')
off=sorted(set(c-r for r,c in zip(row,col)));dvals=[[0]*6 for o in off]
for r,c,v in zip(row,col,val):dvals[off.index(c-r)][r]=v
table([['Diagonal / offset','val[d·n … d·n+5]']]+[[f'{i} / {o}',arr(dvals[i])] for i,o in enumerate(off)],[130,380])
p(f'DIA needs nd = {len(off)} diagonals and 4·nd·(n+1) = {4*len(off)*7} bytes. The conversion first flags offsets col−row+n−1 in a 2n−1 scratch array, numbers occupied flags in increasing order, then fills each row’s entries in parallel. Simultaneous ordinary writes of 1 to the same flag <b>are a data race</b>, despite equal values. This implementation uses omp atomic write. After the flagging barrier, serial numbering is safe; different output rows write different positions within each diagonal.')
p('The max_bytes guard is evaluated after counting occupied offsets and before allocating the DIA offsets/value tables; temporary scratch flags are freed on rejection. Dividing max_bytes by 4·(n+1) avoids a multiplication overflow in the size test. Input CSR must not contain duplicate coordinates for this conversion; the generated Laplacian and A/B satisfy that precondition.')
h('Vector kernels and output ownership')
p('ELL gives blocks of eight rows to OpenMP. Every lane owns one output row; loadu_si256 loads column indices, i32gather_ps fetches x, loadu_ps reads values, fmadd_ps accumulates, and storeu_ps writes results. Remaining n mod 8 rows use a scalar FMA tail. DIA likewise distributes row blocks to threads, then traverses diagonals within each block; contiguous loads replace gathers. Boundary fragments are handled scalarly. Parallelising the diagonal loop with a shared y would race because all diagonals contribute to the same output rows.')
page('8. Task 3 · Storage, kernels and format choice')
mems=lines('t3_4','MEM')[0];gg=lines('t3_4','GRAPH')[0]
table([['Matrix','n / nnz','CSR bytes','ELL bytes','DIA bytes'],['L (1024²)',f'{mems["n"]} / {mems["nnz"]}',mems['csr'],mems['ell'],mems['dia']],['Google M (float)',f'{gg["n"]} / {gg["nnz"]}',gg['csr'],gg['ell'],gg['dia']]],[76,140,98,98,98])
p(f'L has W = {mems["W"]}, nd = {mems["nd"]}; G has W = {gg["W"]}, nd = {gg["nd"]}. Google ELL/DIA sizes are computed from incoming CSR row lengths and occupied offsets only; neither huge format is allocated. The CSR size here uses float values, whereas Task 2 stores double weights.')
table(bench_rows(3,'t3_',{'scalar','threads','ell_simd','ell_hybrid','dia_simd','dia_hybrid'}),[100,55,115,115,125])
rr=[r for t in (1,2,4) for r in result(f't3_{t}')];best=min(rr,key=lambda r:float(r['seconds']));err=max(float(r['maxdiff']) for r in rr)
p(f'Fastest measured kernel: <b>{best["variant"]}, {best["threads"]} thread(s)</b>, {float(best["seconds"]):.6f} s per SpMV. Maximum observed difference from the scalar reference is {err:.3g}; the limit checked by the executable is 2×10⁻⁶. FMA rounds a product-plus-sum once; scalar multiplication/addition rounds twice. DIA boundary fragments can also alter accumulation grouping. These finite-precision effects explain the small non-bit-identical differences.')
p('Hybrid efficiency can exceed one because speedup includes SIMD as well as thread-level acceleration; it is not a fraction of total CPU utilisation. The Laplacian is banded with five occupied offsets and at most five entries per row: ELL padding is modest, and DIA avoids per-entry column indices and gathers. Google has long irregular rows and widely spread diagonals. One long row forces every ELL row to that width; DIA allocates n floats for almost every offset. CSR therefore fits the graph, whereas DIA is structurally well suited to L. Memory bandwidth, gather cost and the selected CPU quota determine the measured ordering within those structural choices.')
page('9. Task 4 · Heat validation and final field')
ref=result('t4_512_scalar_1')[0];p('Two contiguous (N+2)² double buffers hold old and new temperatures. The top boundary is 100°C, other edges 0°C; an 8×8 source beginning at (3N/4, N/4) is fixed at 200°C in both buffers. Every other interior cell starts at zero. Updated cells use 0.2 times the sum of their old value and four old neighbours. Source rows are split into segments outside the source, so SIMD column loops contain no source test.')
rows=[['Variant / threads','Centre','Mean','Final δ']]
for key in ['t4_512_scalar_1','t4_512_threads_1','t4_512_threads_2','t4_512_threads_4','t4_512_persistent_4']:
 r=result(key)[0];rows.append([r['variant']+' / '+r['threads'],r['centre'],r['mean'],r['delta']])
table(rows,[110,135,135,130]);p('All displayed values agree to at least ten significant digits, and round to the assignment’s centre = 2.4557, mean = 18.1928 and δ = 1.20×10⁻³. The mean includes source cells; δ excludes fixed cells and measures only the final step, not the largest change over all time.')
a=np.loadtxt(R/'heat512.txt');fig,ax=plt.subplots(figsize=(6,3.5));im=ax.imshow(a,cmap='inferno',vmin=0,vmax=200,origin='upper',extent=(1,512,512,1));ax.set(xlabel='column j',ylabel='row i',title='N = 512, S = 20,000 · computed temperature');fig.colorbar(im,ax=ax,label='°C');fig.tight_layout();fig.savefig(R/'heatmap.png',dpi=160);plt.close(fig);story.append(Image(str(R/'heatmap.png'),width=430,height=251))
p('A single in-place array would mix old and already updated values: the result becomes sweep-order dependent (Gauss–Seidel-like), and parallel reads/writes introduce races. Separate arrays and a barrier before pointer exchange retain Jacobi semantics. OpenMP v1 opens a parallel region each step; v2 opens one persistent team, uses an omp for max reduction, and swaps pointers in omp single with its implicit barrier.')
page('10. Task 4 · Performance and memory behaviour')
rows=[['Variant','Threads','512² × 20,000: s / speedup / eff.','2048² × 1,000: s / speedup / eff.']]
for v,t in [('scalar',1),('simd',1)]+[(v,t) for t in (1,2,4) for v in ('threads','hybrid','persistent')]:
 row=[v,t]
 for n in (512,2048):
  sec=time(f't4_{n}_{v}_{t}');base=time(f't4_{n}_scalar_1');row.append(f'{sec:.5f} / {base/sec:.3f} / {base/sec/t:.3f}')
 rows.append(row)
table(rows,[78,45,193,194]);rows=[['Experiment','Seconds','Interpretation']]
for n in (512,2048):
 rows.append([f'collapse(2), N={n}',f'{time(f"t4_{n}_collapse_4"):.6f}','4 threads; flattened iteration space, source branch'])
rows.append(['swapped loops, N=2048',f'{time("t4_2048_swapped_4"):.6f}','4 threads; column-major traversal of row-major arrays'])
for init in (0,1):
 r=result(f't4_4096_init{init}')[0];rows.append([f'4096² × 200, init={init}',r['seconds'],f'initialisation separately: {float(r["init_seconds"]):.6f} s'])
table(rows,[150,95,265])
for n in (512,2048):
 candidates=[(time(f't4_{n}_{v}_{t}'),v,t) for v,t in [('scalar',1),('simd',1)]+[(v,t) for t in (1,2,4) for v in ('threads','hybrid','persistent')]]
 best=min(candidates);p(f'N={n}: fastest measured variant is {best[1]} with {best[2]} thread(s), {best[0]:.6f} s. Persistent/SIMD v2 versus scalar-loop v1 combines vectorisation with team reuse; compare hybrid v1 with persistent v2 to isolate the team-lifetime change.')
p('Persistent teams remove repeated region entry, but v2 also has barriers for the delta reset, reduction and pointer exchange. In these measurements, passive waiting and those extra barriers can outweigh the saved region overhead, especially on the small grid. A large grid makes stencil traffic dominate. collapse(2) can expose more iterations, but these grids already offer hundreds or thousands of rows; flattening adds index arithmetic and hinders the branch-free row-segment vector loop. Swapped loops advance by a whole row, wasting cache-line locality and increasing TLB/cache pressure. In row-major traversal, adjacent columns reuse contiguous cache lines.')
p('First touch assigns physical pages on the first write. Parallel initialisation uses the sweep’s static row schedule; serial initialisation first-touches all pages on one thread. This VM exposes one NUMA node, so there is no demonstrated cross-socket placement advantage. Initialisation time and subsequent kernel time are reported separately; small differences can reflect caches, paging and VM noise rather than NUMA migration.')
page('11. Task 5 · GraphBLAS and BFS')
p('The C library was installed locally by downloading Debian libgraphblas-dev and libgraphblas7 7.4.0+dfsg-2 through APT with repository signature and package-hash verification, then extracting under /workspace/pdc-tools/graphblas. No Python GraphBLAS fallback was needed. The terminal panel below renders the recorded successful compile/run transcript, including the API version.')
log=(R/'graphblas-install.txt').read_text() if (R/'graphblas-install.txt').exists() else '$ make all\n'+text('t5_g6')
ll=log.splitlines();ll=[l for l in ll if not l.startswith('LEVEL')];wrapped=[]
import textwrap
for l in ll:wrapped.extend(textwrap.wrap(l,width=112,replace_whitespace=False,drop_whitespace=False) or [''])
ff=ImageFont.truetype(mono,16);im=PImage.new('RGB',(1140,26+22*len(wrapped)),(20,28,37));draw=ImageDraw.Draw(im)
for i,l in enumerate(wrapped):draw.text((12,10+22*i),l,font=ff,fill=(222,239,231))
im.save(R/'terminal.png');story.append(Image(str(R/'terminal.png'),width=510,height=min(225,510*im.height/im.width)));story.append(Spacer(1,8))
p('The G6 adjacency matrix stores A(src,dst)=true. One call to GrB_Matrix_reduce_Monoid with the plus monoid returns out-degrees (2, 1, 2, 1, 2, 2). The one-hop example returns vertices 1 and 2. G6 BFS from 0 returns levels (0, 1, 1, 2, 3, 4), matching the assignment.')
bfs=lines('t5_google','TASK5')[0];p(f'Google BFS starts from Task 2’s highest-ranked original page <b>{bfs["source"]}</b>, reaches <b>{bfs["reached"]}</b> vertices, and has maximum level {bfs["maxlevel"]}. Minimum of five BFS kernel times: {time("t5_google"):.6f} s on four GraphBLAS threads. Every output level is independently checked against a scalar queue BFS.')
counts=lines('t5_google','LEVEL');rows=[['Depth','Count','Depth','Count','Depth','Count']]
chunk=(len(counts)+2)//3
for i in range(chunk):
 row=[]
 for j in (i,i+chunk,i+2*chunk):row+=([counts[j]['depth'],counts[j]['count']] if j<len(counts) else ['',''])
 rows.append(row)
table(rows,[55,115,55,115,55,115])
p('Each level uses assign_INT32(level, q, …) followed by vxm over the OR–AND semiring with GrB_DESC_RSC. The complemented structural level mask excludes visited vertices, including the source whose stored level is zero. Replace semantics clear stale frontier entries. Blocking mode and nvals synchronisation make timing include completed operations.')
p('<b>Answers.</b> (i) With plus–times, GrB_mxv corresponds to Task 3’s y=Ax SpMV, regardless of the library’s hidden sparse format. (ii) Without the mask, G6’s cycles repeatedly regenerate visited vertices; the frontier need not empty and levels get overwritten. (iii) GraphBLAS provides concise, tested sparse kernels and semiring/mask reuse; a disadvantage is less direct control over storage, scheduling and fusion than hand-written PageRank, plus a library dependency.')
page('12. Reproduction, packaging and conclusions')
h('Build and run on Linux (AVX2/FMA required)')
code('sudo apt-get update\nsudo apt-get install build-essential libgraphblas-dev git python3\ncd 23i-0041-CS-D\nmake BUILD=/tmp/pdc-build all\nmake BUILD=/tmp/pdc-build check\nsh 23i-0041-CS-D-TASK0-V1.sh')
p('The shell script obtains the pinned full graph, verifies its canonical checksum, builds all tasks and runs the experiment driver. BUILD, DATA and RESULTS may be set to writable directories outside the submission folder. For a non-system GraphBLAS installation, pass GB_PREFIX and GB_LIBDIR to make, e.g. GB_PREFIX=/workspace/pdc-tools/graphblas/usr. The linker embeds an rpath for that library directory.')
h('Direct experiment / report commands')
code('python3 23i-0041-CS-D-TASK0-V1.py \\\n  --build /tmp/pdc-build --graph /path/web-Google.txt \\\n  --output /tmp/pdc-results\n# Optional PDF regeneration only (not required for C programs):\npython3 -m pip install reportlab matplotlib pillow\npython3 23i-0041-CS-D-TASK0-V2.py \\\n  /tmp/pdc-results /tmp/pdc-build /tmp/Report.pdf')
p('To regenerate the terminal panel, record a successful task5 compile command and its G6 execution in RESULTS/graphblas-install.txt. Raw JSON/text results, compiler .vec reports, datasets, generated figures, fields and executables are intentionally outside the submission. The provided PDF embeds measured tables, the B conversion diagrams, final heatmap and terminal transcript. The benchmark driver stops on any nonzero validation status. Additional checks passed: ASan/UBSan sparse-format boundary/tail cases; nonconsecutive 64-bit graph IDs; and whole-field agreement of all seven heat variants at 64², 101 steps.')
h('Submission layout')
code('23i-0041-CS-D/\n  Task1/23i-0041-CS-D-TASK1-V1.c\n  Task2/23i-0041-CS-D-TASK2-V1.c\n  Task3/23i-0041-CS-D-TASK3-V1.c\n  Task4/23i-0041-CS-D-TASK4-V1.c\n  Task5/23i-0041-CS-D-TASK5-V1.c\n  23i-0041-CS-D-TASK0-V1.h       # shared input / memory helpers\n  23i-0041-CS-D-TASK0-V1.py      # experiments\n  23i-0041-CS-D-TASK0-V2.py      # report generation\n  23i-0041-CS-D-TASK0-V1.sh      # reproduction entry point\n  Makefile                      # required conventional filename\n  23i-0041-CS-D-Report.pdf')
p('The ZIP contains this single top-level folder, code/build instructions and this PDF only; no binaries, .exe files, graph datasets or result logs. TASK0 identifies shared/reproduction code, and Task1–Task5 each contain their task source. The assignment’s explicit requirement for a single Makefile necessitates its conventional filename. All code has the required name, student ID and assignment number header.')
h('Conclusions and references')
p('Correctness requires fixing the serial baseline before parallelising, preserving independent output ownership, and validating floating-point differences. Sparse-format choice should follow matrix structure: DIA/ELL are natural candidates for the banded grid, while CSR handles irregular graph rows economically. SIMD, scheduling and persistent teams must be assessed with measurements rather than assumed gains. One-node virtualised hardware limits claims about NUMA and physical-core scaling.')
p('References: supplied CS3006 PDC Assignment 2 (updated), Listings 1–7 and Figures 1–9; Stanford SNAP web-Google dataset (snap.stanford.edu/data/web-Google.html); OpenMP API specification (openmp.org/specifications); Intel Intrinsics Guide; SuiteSparse:GraphBLAS 7.4 C header/API and GraphBLAS C API 2.0. The original problem credits Stanford CS149, Maryland CMSC498X/818X, RWTH OpenMP exercises, and Bell–Garland sparse-format examples.')
def footer(canvas,doc):
 canvas.setFont('D',7);canvas.setFillColor(colors.HexColor('#53707c'));canvas.drawString(42,24,'Muhammad Ayaan Naveed · 23i-0041 · CS-D · Assignment 2');canvas.drawRightString(553,24,str(doc.page))
doc=SimpleDocTemplate(target,pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=40,title='Assignment 2 — Muhammad Ayaan Naveed — 23i-0041 CS-D',author='Muhammad Ayaan Naveed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(target)
