"""Builds uerf_fewshot_phase.ipynb — the brain's REAL claim: learn accurately
from a HANDFUL of examples, not thousands. Fresh brain, k shots/class, phase
coupling live FROM BIRTH (the low-α regime where cos φ_ij has leverage).

Two questions in one run:
  1. EFFICIENCY: accuracy vs k ∈ {1,2,3,5,10,20} shots/class — how few examples
     does a physics brain need to learn MNIST properly?
  2. PHASE FROM BIRTH: does the Unified-Master-Equation phase coupling help when
     it's present during the formative learning (vs the decohered/classical
     limit from birth)?
Total data at k=5 is 50 images. This is the opposite of brute training.
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
FEW-SHOT FROM SCRATCH — learn from a handful, phase coupling from birth.
"""
import os, time, numpy as np, torch
from uerf_brain import UERFField, SensoryProjection, S_QUANTUM, S_QUBIT
from torchvision import datasets, transforms
to_t=transforms.ToTensor()
dev="cuda"
mn_tr=datasets.MNIST("./d",train=True,download=True); mn_te=datasets.MNIST("./d",train=False,download=True)
tr_imgs=[i for i,_ in mn_tr]; tr_lbl=torch.tensor([l for _,l in mn_tr])
mean_img=torch.stack([to_t(i).view(-1) for i in tr_imgs[:8000]]).mean(0)
proj=SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev)
Xte=torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte=torch.tensor([l for _,l in mn_te])
enc=lambda pil: proj(to_t(pil).view(-1).to(dev))

# fixed few-shot support set: first k examples of each class (interleaved by round)
by_class={d:(tr_lbl==d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
eval_idx=torch.randperm(len(Xte))[:2000].tolist()

def decohere(f):
    f.s_imag=torch.zeros_like(f.s_imag)
    f._quantum_evolve=lambda new_s,alpha:new_s   # classical limit from birth
    return f

def teach(k, phase_on):
    torch.manual_seed(369)
    f=UERFField(n_max=400,n_initial=200,d=32,input_dim=64,n_classes=15)
    f.start_emergent()
    if not phase_on: decohere(f)
    label={}
    # interleave: round r presents one example of each class (avoids recency bias)
    for r in range(k):
        for d in range(10):
            i=by_class[d][r]; x=enc(tr_imgs[i])
            if d not in label:
                sig=(x.unsqueeze(-1)*f.c[:f.input_dim]).sum(0); label[d]=f.birth_category(sig)
            tv=torch.zeros(f.n_classes,device=dev); tv[label[d]]=1.0
            f.experience(x,teaching_vector=tv,n_relax=8,domain_lo=0,domain_hi=f.n_classes)
    return f,label

def evaluate(f,label):
    inv={v:d for d,v in label.items()}; lo,hi=min(label.values()),max(label.values())+1
    c=t=0
    for i in eval_idx:
        p=f.eval_predict(proj(Xte[i].to(dev)),domain_lo=lo,domain_hi=hi)
        c+=int(inv.get(p)==int(Yte[i])); t+=1
    return 100.0*c/t

SHOTS=[1,2,3,5,10,20]
print("="*64); print(" FEW-SHOT FROM SCRATCH — handful of examples, phase from birth"); print("="*64)
print(f"  {'shots':>6}{'imgs':>6}{'PHASE%':>9}{'CLASSIC%':>10}{'Δ':>7}{'coh':>7}{'cats':>6}")
res=[]
for k in SHOTS:
    fp,lp=teach(k,True);  ap=evaluate(fp,lp)
    coh=float(fp.s_imag.abs().sum())
    nq=int((fp.state_id==S_QUANTUM).sum()+(fp.state_id==S_QUBIT).sum())
    fc,lc=teach(k,False); ac=evaluate(fc,lc)
    print(f"  {k:>6}{k*10:>6}{ap:>9.1f}{ac:>10.1f}{ap-ac:>+7.1f}{coh:>7.2f}{len(lp):>6}",flush=True)
    res.append((k,ap,ac))

print("\n"+"="*64); print(" READ"); print("="*64)
best_k,best_a,_=max(res,key=lambda r:r[1])
one=[r for r in res if r[0]==1][0]; five=[r for r in res if r[0]==5][0]
print(f"  1 shot/class  (10 imgs total): phase {one[1]:.1f}%")
print(f"  5 shots/class (50 imgs total): phase {five[1]:.1f}%")
print(f"  best: {best_a:.1f}% at {best_k} shots/class ({best_k*10} images)")
dmax=max(r[1]-r[2] for r in res); davg=sum(r[1]-r[2] for r in res)/len(res)
print(f"  phase vs classical: avg Δ {davg:+.1f}%, max Δ {dmax:+.1f}%")
if davg>1.0: print("  => phase-from-birth helps few-shot (framework cos φ_ij pays off when α is low)")
elif davg<-1.0: print("  => phase-from-birth hurts; classical imprinting cleaner")
else: print("  => phase neutral even from birth; efficiency is the story")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile fs.py\n"+HARNESS,"harness"),cell("!python fs.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_fewshot_phase.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
