"""Builds uerf_honest.ipynb — the TRUE results on the cleaned, un-cheated brain.

Finished script: working associative-memory readout, ALL banned supervised
readouts deleted, ONLINE self-computed centering (no offline dataset mean),
argmax over born categories only (no domain-narrowing crutch). One fresh brain,
single online pass (each example once). We report exactly what the brain earns:

  • EFFICIENCY   — accuracy vs shots (10..200 images total), online single pass
  • ONE-SHOT     — after 1 example/class
  • CONTINUAL    — learn 0-4, then 5-9, re-probe 0-4 over ALL born classes
                   (honest: no domain mask hiding interference)
  • SURPRISE     — mean surprise falling as concepts consolidate

Honest framing: the classifier is Hebbian associative memory (percept->name);
the oscillator physics is the brain's internal life, measured on its own terms,
not credited with the accuracy.
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

HARNESS = r'''"""TRUE RESULTS — cleaned brain, online centering, no crutches."""
import numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms
to_t=transforms.ToTensor(); dev="cuda"
mn_tr=datasets.MNIST("./d",train=True,download=True); mn_te=datasets.MNIST("./d",train=False,download=True)
tr_imgs=[i for i,_ in mn_tr]; tr_lbl=torch.tensor([l for _,l in mn_tr])
Xte=torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte=torch.tensor([l for _,l in mn_te])
by_class={d:(tr_lbl==d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
PROBE=torch.randperm(len(Xte))[:800].tolist()

def fresh_proj():
    # ONLINE centering — the brain computes its own DC estimate. No dataset mean.
    return SensoryProjection(784,64,seed=42,online=True,ema=0.03).to(dev)

def probe(field,proj_,label,classes=None):
    inv={v:d for d,v in label.items()}
    lo,hi=min(label.values()),max(label.values())+1        # over BORN categories only
    c=t=0
    for i in PROBE:
        y=int(Yte[i])
        if classes is not None and y not in classes: continue
        p=field.eval_predict(proj_(Xte[i].to(dev)),domain_lo=lo,domain_hi=hi)  # Xte already flat
        c+=int(inv.get(p)==y); t+=1
    return 100.0*c/max(t,1)

def fresh():
    torch.manual_seed(369)
    f=UERFField(n_max=200,n_initial=120,d=32,input_dim=64,n_classes=15); f.start_emergent()
    return f

# 1. EFFICIENCY — single online pass, honest online centering
print("="*64); print(" TRUE RESULTS — cleaned brain, online centering, no crutches"); print("="*64)
f=fresh(); proj=fresh_proj(); enc=lambda t: proj(t.to(dev) if torch.is_tensor(t) else t)
encimg=lambda pil: proj(to_t(pil).view(-1).to(dev))
label={}; sched={10,20,30,50,80,120,160,200}; sur=[]; curve=[]
order=[(d,by_class[d][r]) for r in range(20) for d in range(10)]
for step,(d,i) in enumerate(order,1):
    x=encimg(tr_imgs[i])
    if d not in label:
        sig=(x.unsqueeze(-1)*f.c[:f.input_dim]).sum(0); label[d]=f.birth_category(sig)
    tv=torch.zeros(f.n_classes,device=dev); tv[label[d]]=1.0
    f.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=f.n_classes)
    sur.append(float(getattr(f,"_input_surprise",0.0)))
    if step in sched:
        a=probe(f,proj,label)
        curve.append((step,a))
        print(f"  seen {step:>4} ({step//10:>2} shots/class)  acc {a:5.1f}%   surprise {np.mean(sur[-10:]):.2f}",flush=True)
one=[a for s,a in curve if s==10][0]

# 2. CONTINUAL — honest, argmax over ALL born classes (no domain mask)
f2=fresh(); proj2=fresh_proj(); enc2=lambda pil: proj2(to_t(pil).view(-1).to(dev))
label2={}
def teach(classes,shots=8):
    for r in range(shots):
        for d in classes:
            x=enc2(tr_imgs[by_class[d][r]])
            if d not in label2:
                sig=(x.unsqueeze(-1)*f2.c[:f2.input_dim]).sum(0); label2[d]=f2.birth_category(sig)
            tv=torch.zeros(f2.n_classes,device=dev); tv[label2[d]]=1.0
            f2.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=f2.n_classes)
teach(range(0,5)); A_before=probe(f2,proj2,label2,classes=set(range(5)))
teach(range(5,10)); A_after=probe(f2,proj2,label2,classes=set(range(5))); B=probe(f2,proj2,label2,classes=set(range(5,10)))

print("\n"+"="*64); print(" TRUE REPORT (no crutches)"); print("="*64)
print(f"  one-shot (1 ex/class, 10 imgs)   : {one:.1f}%   (chance 10%)")
best=max(curve,key=lambda x:x[1])
print(f"  best single online pass          : {best[1]:.1f}% at {best[0]} imgs ({best[0]//10} shots/class)")
print(f"  surprise decayed                 : {np.mean(sur[:10]):.2f} -> {np.mean(sur[-10:]):.2f}")
print(f"  continual 0-4 before / after 5-9 : {A_before:.1f}% -> {A_after:.1f}%   (forgetting {A_before-A_after:+.1f})")
print(f"  newly-learned 5-9                : {B:.1f}%")
print("\n  Classifier = Hebbian associative memory (percept->name).")
print("  Physics (states/coherence/vacuum) = internal life, not credited with accuracy.")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile h.py\n"+HARNESS,"harness"),cell("!python h.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_honest.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
