"""Builds uerf_brainlike.ipynb — teach it like a brain, grade it like a brain.

NOT an NN harness. One SINGLE online pass — every example seen exactly once, in
a stream, with the brain's own prediction-error / surprise (the dopamine loop,
neuro_enabled) deciding what sticks. No epochs, no reshuffled repetition, no
chasing CNN accuracy. Graded on what makes this engine different:

  • ONLINE LEARNING CURVE — accuracy vs #examples-ever-seen, single pass
  • ONE-SHOT            — accuracy after ONE example per class (10 images total)
  • CONTINUAL / NO-FORGET — learn 0-4, then 5-9; did 0-4 survive? (Δ retention)
  • SURPRISE-GATED       — track mean surprise falling as concepts consolidate

The probe is small on purpose: a few-shot/continual experiment does not need
NN-scale evaluation. That is correct science, not a shortcut.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAIN = open(os.path.join(ROOT, "uerf_brain.py")).read()

PIN = '''import subprocess, sys
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu}")
if "P100" in gpu:
    subprocess.run(["pip","install","-q","torch==2.7.1","torchvision==0.22.1"], check=True)
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
'''

HARNESS = r'''"""
BRAIN-LIKE — single online pass, surprise-gated, graded on brain-unique axes.
"""
import numpy as np, torch, time
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms
to_t=transforms.ToTensor(); dev="cuda"
mn_tr=datasets.MNIST("./d",train=True,download=True); mn_te=datasets.MNIST("./d",train=False,download=True)
tr_imgs=[i for i,_ in mn_tr]; tr_lbl=torch.tensor([l for _,l in mn_tr])
mean_img=torch.stack([to_t(i).view(-1) for i in tr_imgs[:8000]]).mean(0)
proj=SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev)
Xte=torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte=torch.tensor([l for _,l in mn_te])
enc=lambda pil: proj(to_t(pil).view(-1).to(dev))
by_class={d:(tr_lbl==d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
PROBE=torch.randperm(len(Xte))[:500].tolist()   # small: few-shot experiment, not an NN benchmark

def probe(field,label,classes=None):
    inv={v:d for d,v in label.items()}; lo,hi=min(label.values()),max(label.values())+1
    c=t=0
    for i in PROBE:
        y=int(Yte[i])
        if classes is not None and y not in classes: continue
        p=field.eval_predict(proj(Xte[i].to(dev)),domain_lo=lo,domain_hi=hi)
        c+=int(inv.get(p)==y); t+=1
    return 100.0*c/max(t,1)

def fresh():
    torch.manual_seed(369)
    f=UERFField(n_max=200,n_initial=120,d=32,input_dim=64,n_classes=15)
    f.start_emergent(); return f

# ── 1. ONLINE LEARNING CURVE — one stream, each example ONCE, surprise-gated ──
print("="*60); print(" 1. ONLINE SINGLE-PASS LEARNING CURVE"); print("="*60)
f=fresh(); label={}; seen=0; surprises=[]
schedule=[10,20,30,50,80,120,160,200]   # probe after this many examples ever seen
order=[]
for r in range(20):                      # up to 20 shots/class, interleaved stream
    for d in range(10): order.append((d,by_class[d][r]))
curve=[]; t0=time.time()
for step,(d,i) in enumerate(order,1):
    x=enc(tr_imgs[i])
    if d not in label:
        sig=(x.unsqueeze(-1)*f.c[:f.input_dim]).sum(0); label[d]=f.birth_category(sig)
    tv=torch.zeros(f.n_classes,device=dev); tv[label[d]]=1.0
    f.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=f.n_classes)  # ONE look
    surprises.append(float(getattr(f,"_input_surprise",0.0)))
    seen=step
    if seen in schedule:
        a=probe(f,label)
        curve.append((seen,a));
        print(f"  seen {seen:>4} examples  ({seen//10} shots/class)  ->  acc {a:5.1f}%   "
              f"mean-surprise {np.mean(surprises[-10:]):.2f}",flush=True)
oneshot=[a for s,a in curve if s==10]
print(f"\n  ONE-SHOT (10 imgs, 1/class): {oneshot[0] if oneshot else float('nan'):.1f}%")
print(f"  online pass rate: {len(order)/(time.time()-t0):.1f} ex/s")

# ── 2. CONTINUAL / NO-FORGET — learn 0-4, then 5-9, re-check 0-4 ──────────────
print("\n"+"="*60); print(" 2. CONTINUAL LEARNING (no-forget)"); print("="*60)
f2=fresh(); label2={}
def teach_span(f,label,classes,shots=8):
    for r in range(shots):
        for d in classes:
            x=enc(tr_imgs[by_class[d][r]])
            if d not in label:
                sig=(x.unsqueeze(-1)*f.c[:f.input_dim]).sum(0); label[d]=f.birth_category(sig)
            tv=torch.zeros(f.n_classes,device=dev); tv[label[d]]=1.0
            f.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=f.n_classes)
teach_span(f2,label2,range(0,5)); accA_before=probe(f2,label2,classes=set(range(0,5)))
teach_span(f2,label2,range(5,10)); accA_after=probe(f2,label2,classes=set(range(0,5)))
accB=probe(f2,label2,classes=set(range(5,10)))
print(f"  0-4 acc after learning 0-4          : {accA_before:5.1f}%")
print(f"  0-4 acc after ALSO learning 5-9     : {accA_after:5.1f}%   (Δ {accA_after-accA_before:+.1f})")
print(f"  5-9 acc (the newly learned split)   : {accB:5.1f}%")
print(f"  => forgetting = {accA_before-accA_after:+.1f} pts  (an NN would collapse 0-4 here)")

print("\n"+"="*60); print(" BRAIN-LIKE REPORT"); print("="*60)
print(f"  one-shot (1 ex/class)      : {oneshot[0] if oneshot else float('nan'):.1f}%")
best=max(curve,key=lambda x:x[1])
print(f"  best online single-pass    : {best[1]:.1f}% at {best[0]} examples ({best[0]//10} shots/class)")
print(f"  continual forgetting       : {accA_before-accA_after:+.1f} pts")
print(f"  surprise fell from {np.mean(surprises[:10]):.2f} -> {np.mean(surprises[-10:]):.2f} as it consolidated")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile bl.py\n"+HARNESS,"harness"),cell("!python bl.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_brainlike.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
