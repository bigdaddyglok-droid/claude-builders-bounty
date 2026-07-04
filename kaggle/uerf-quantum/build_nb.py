"""Builds uerf_quantum.ipynb — A/B: literal quantum states vs classical stand-in."""
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
LITERAL QUANTUM A/B — does genuine quantum evolution help?
Two identical brains, same seed, same data. One runs the LITERAL quantum states
(unitary + Lindblad on QUANTUM/QUBIT nodes); the CONTROL has its _quantum_evolve
neutralized to a no-op (the old scalar-only behavior). We compare accuracy,
retention, and coherence. The brain itself has no toggle — the control is made
by swapping the method in the harness, so the physics stays the only path.
"""
import os, time, numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms

dev="cuda"
to_t=transforms.ToTensor()
tr=datasets.MNIST("./d",train=True,download=True); te=datasets.MNIST("./d",train=False,download=True)
Xtr=torch.stack([to_t(i).view(-1) for i,_ in tr]); Ytr=torch.tensor([l for _,l in tr])
Xte=torch.stack([to_t(i).view(-1) for i,_ in te]); Yte=torch.tensor([l for _,l in te])
mean_img=Xtr.mean(0)
proj=SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev)
A=[0,1,2,3,4]; B=[5,6,7,8,9]
def poolof(cs):
    m=torch.zeros(len(Ytr),dtype=torch.bool)
    for c in cs: m|=(Ytr==c)
    return m.nonzero(as_tuple=True)[0].tolist()
poolA,poolB=poolof(A),poolof(B)

STEPS=1000
def make(quantum):
    torch.manual_seed(369)
    f=UERFField(n_max=400,n_initial=200,d=32,input_dim=64,n_classes=15)
    f.start_emergent()
    if not quantum:
        f._quantum_evolve = lambda new_s, alpha: new_s  # neutralize (instance attr, no self)
    return f

def run(field, label, tag):
    rng=np.random.RandomState(7); t0=time.time()
    def teach(pool,n):
        for step in range(n):
            i=pool[rng.randint(0,len(pool))]; y=int(Ytr[i]); x=proj(Xtr[i].to(dev))
            if y not in label:
                sig=(x.unsqueeze(-1)*field.c[:field.input_dim]).sum(0); label[y]=field.birth_category(sig)
            tv=torch.zeros(field.n_classes,device=dev); tv[label[y]]=1.0
            field.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=field.n_classes)
    def acc(classes,n=400):
        cats=[label[c] for c in classes if c in label]
        if not cats: return 0.0
        lo,hi=min(cats),max(cats)+1; inv={v:k for k,v in label.items()}
        idx=torch.randperm(len(Xte))[:n*3].tolist(); c=t=0
        for i in idx:
            if int(Yte[i]) not in classes: continue
            if t>=n: break
            p=field.eval_predict(proj(Xte[i].to(dev)),domain_lo=lo,domain_hi=hi)
            c+=int(inv.get(p)==int(Yte[i])); t+=1
        return 100.0*c/max(t,1)
    print(f"\n[{tag}] training A then B ({STEPS} each)...",flush=True)
    teach(poolA,STEPS); accA_pre=acc(A)
    teach(poolB,STEPS); accA_post=acc(A); accB=acc(B); accAll=acc(A+B)
    r=field.report()
    coh=float(field.s_imag.abs().sum().item())
    nq=r['state_dist'].get('QUANTUM',0)+r['state_dist'].get('QUBIT',0)
    print(f"[{tag}] A {accA_pre:.1f}->{accA_post:.1f} | B {accB:.1f} | all {accAll:.1f} | "
          f"Q/QUBIT nodes={nq} coherence={coh:.2f}  ({(2*STEPS)/(time.time()-t0):.1f}/s)",flush=True)
    return dict(accA_pre=accA_pre,accA_post=accA_post,accB=accB,accAll=accAll,nq=nq,coherence=coh)

print("="*60); print(" LITERAL QUANTUM vs CLASSICAL STAND-IN"); print("="*60)
ctrl = run(make(False), {}, "CONTROL (no quantum evolution)")
quant= run(make(True),  {}, "QUANTUM (literal unitary+Lindblad)")

print("\n"+"="*60); print(" VERDICT"); print("="*60)
print(f"  {'metric':16}{'CONTROL':>10}{'QUANTUM':>10}{'Δ':>8}")
for k in ['accA_pre','accA_post','accB','accAll']:
    print(f"  {k:16}{ctrl[k]:>10.1f}{quant[k]:>10.1f}{quant[k]-ctrl[k]:>+8.1f}")
print(f"  {'retention':16}{ctrl['accA_post']-ctrl['accA_pre']:>+10.1f}{quant['accA_post']-quant['accA_pre']:>+10.1f}")
print(f"  quantum coherence present only in QUANTUM brain: {quant['coherence']:.2f} vs {ctrl['coherence']:.2f}")
d=quant['accAll']-ctrl['accAll']
if d>1.5:   print(f"\n  => quantum HELPS (+{d:.1f} all-10)")
elif d<-1.5:print(f"\n  => quantum HURTS ({d:.1f} all-10)")
else:       print(f"\n  => NEUTRAL on accuracy ({d:+.1f}); coherence exists but no measurable task gain yet")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile q.py\n"+HARNESS,"harness"),cell("!python q.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_quantum.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
