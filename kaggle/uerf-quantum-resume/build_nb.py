"""Builds uerf_quantum_resume.ipynb — resume trained brain, A/B quantum at inference."""
import json, os

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
QUANTUM AT INFERENCE (resume, no retrain) — does quantum evolution change the
readout of an already-trained brain? We load consolidated_brain.pt and evaluate
the SAME brain twice: once with literal _quantum_evolve active, once neutralized
to a no-op. Difference = what quantum coherence does at inference time. Minutes,
because it resumes instead of training.
"""
import os, sys, glob, numpy as np, torch
def find(name):
    h=sorted(glob.glob(f"/kaggle/input/**/{name}",recursive=True),key=len)
    if not h: raise FileNotFoundError(name)
    return h[0]
BRAIN=find("uerf_brain.py"); CKPT=find("consolidated_brain.pt")
sys.path.insert(0,os.path.dirname(BRAIN))
import importlib.util
spec=importlib.util.spec_from_file_location("uerf_brain",BRAIN)
U=importlib.util.module_from_spec(spec); sys.modules["uerf_brain"]=U; spec.loader.exec_module(U)
dev="cuda"
from torchvision import datasets, transforms
to_t=transforms.ToTensor()
mn_tr=datasets.MNIST("./d",train=True,download=True); mn_te=datasets.MNIST("./d",train=False,download=True)
mean_img=torch.stack([to_t(i).view(-1) for i,_ in mn_tr]).mean(0)
proj=U.SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev)
Xte=torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte=torch.tensor([l for _,l in mn_te])

def load():
    return U.UERFField.load_checkpoint(CKPT, device=dev)

def digitmap(field):
    from collections import Counter
    dc={}
    for d in range(10):
        idx=(Yte==d).nonzero(as_tuple=True)[0][:60].tolist()
        v=Counter(field.eval_predict(proj(Xte[i].to(dev)),domain_lo=0,domain_hi=field.n_classes) for i in idx)
        dc[d]=v.most_common(1)[0][0]
    return dc

def acc(field, dc, idx):
    inv={v:k for k,v in dc.items()}; lo,hi=min(dc.values()),max(dc.values())+1
    c=t=0
    for i in idx:
        p=field.eval_predict(proj(Xte[i].to(dev)),domain_lo=lo,domain_hi=hi)
        c+=int(inv.get(p)==int(Yte[i])); t+=1
    return 100.0*c/max(t,1)

idx=torch.randperm(len(Xte))[:1500].tolist()

# ── QUANTUM ON (literal) ─────────────────────────────────────────────────────
fq=load()
nq=(fq.state_id==U.S_QUANTUM).sum().item()+(fq.state_id==U.S_QUBIT).sum().item()
print(f"[resume] loaded brain, {fq._active_categories} categories, "
      f"{nq} QUANTUM/QUBIT nodes",flush=True)
dcq=digitmap(fq)
accQ=acc(fq,dcq,idx)
cohQ=float(fq.s_imag.abs().sum().item())   # coherence accrued during eval

# ── QUANTUM OFF (neutralized) ────────────────────────────────────────────────
fc=load()
fc._quantum_evolve=lambda new_s,alpha:new_s   # no-op
dcc=digitmap(fc)
accC=acc(fc,dcc,idx)
cohC=float(fc.s_imag.abs().sum().item())

print("\n"+"="*60); print(" QUANTUM AT INFERENCE (resumed brain, no retrain)"); print("="*60)
print(f"  QUANTUM/QUBIT nodes in the trained brain : {nq}")
print(f"  accuracy  quantum-ON  : {accQ:.2f}%   coherence accrued: {cohQ:.2f}")
print(f"  accuracy  quantum-OFF : {accC:.2f}%   coherence accrued: {cohC:.2f}")
d=accQ-accC
print(f"  Δ (quantum − classical readout): {d:+.2f}%")
if d>1.0:   print("  => quantum evolution HELPS the readout")
elif d<-1.0:print("  => quantum evolution HURTS the readout")
else:       print("  => NEUTRAL at inference (coherence present, no readout change)")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell(HARNESS,"harness")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_quantum_resume.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out)
