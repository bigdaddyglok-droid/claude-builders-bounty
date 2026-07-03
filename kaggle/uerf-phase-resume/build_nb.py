"""Builds uerf_phase_resume.ipynb — phase-angle cross-state coupling A/B, RESUME.

Loads the already-trained consolidated brain (no from-scratch retrain) and asks
whether the Unified Master Equation's PHASE ANGLE term cos(phi_ij) — now live in
_dynamics_step's cross-state coupling — changes anything:
  (A) at pure inference (expected ~null: trained brain sits at high alpha, no
      phase accrues), and
  (B) after a SHORT continuation (300 steps) where alpha is lower so coherence
      builds and the phase can interfere.
Control arm = the decohered/classical limit (s_imag=0, quantum evolve no-op ->
cos=1 -> plain diffusion). Not a code toggle: it's the field prepared classical.
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
PHASE-COUPLING A/B (resume) — does cos(phi_ij) in the cross-state coupling help?
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
tr_imgs=[i for i,_ in mn_tr]; tr_lbl=torch.tensor([l for _,l in mn_tr])
mean_img=torch.stack([to_t(i).view(-1) for i in tr_imgs[:8000]]).mean(0)
proj=U.SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev)
Xte=torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte=torch.tensor([l for _,l in mn_te])

def load(): return U.UERFField.load_checkpoint(CKPT, device=dev)

def decohere(f):
    """Prepare the field in the classical limit: no accrued phase, quantum
    evolution reduced to identity. cos(phi)=1 -> plain energy diffusion."""
    f.s_imag = torch.zeros_like(f.s_imag)
    f._quantum_evolve = lambda new_s, alpha: new_s      # instance attr -> 2-arg call, no self
    return f

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

def continue_train(field, dc, steps=300):
    """Short reinforcement of the EXISTING category->digit map (no new cats)."""
    cat_for_digit={d:c for d,c in dc.items()}
    rng=np.random.RandomState(11)
    for _ in range(steps):
        i=int(rng.randint(0,len(tr_imgs))); y=int(tr_lbl[i])
        cat=cat_for_digit.get(y)
        if cat is None: continue
        x=proj(to_t(tr_imgs[i]).view(-1).to(dev))
        tv=torch.zeros(field.n_classes,device=dev); tv[cat]=1.0
        field.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=field.n_classes)
    return field

idx=torch.randperm(len(Xte))[:1500].tolist()
print("="*60); print(" PHASE-ANGLE COUPLING A/B (resume, no from-scratch train)"); print("="*60)

# ── (A) PURE INFERENCE ────────────────────────────────────────────────────────
fp=load();  dcp=digitmap(fp);  accP_inf=acc(fp,dcp,idx);  cohP=float(fp.s_imag.abs().sum())
fc=decohere(load()); dcc=digitmap(fc); accC_inf=acc(fc,dcc,idx)
nq=(load().state_id==U.S_QUANTUM).sum().item()+(load().state_id==U.S_QUBIT).sum().item()
print(f"\n[A] pure inference  (Q/QUBIT nodes={nq})")
print(f"    phase-coupled : {accP_inf:.2f}%   coherence={cohP:.3f}")
print(f"    classical     : {accC_inf:.2f}%")
print(f"    Δ (phase − classical) = {accP_inf-accC_inf:+.2f}%")

# ── (B) SHORT CONTINUATION (leverage trained brain, no brute retrain) ─────────
fp2=load();  dcp2=digitmap(fp2)
fp2=continue_train(fp2, dcp2, steps=300); dcp2b=digitmap(fp2)
accP_ft=acc(fp2,dcp2b,idx); cohP2=float(fp2.s_imag.abs().sum())
fc2=decohere(load()); dcc2=digitmap(fc2)
fc2=continue_train(fc2, dcc2, steps=300); dcc2b=digitmap(fc2)
accC_ft=acc(fc2,dcc2b,idx)
print(f"\n[B] +300-step continuation")
print(f"    phase-coupled : {accP_ft:.2f}%   coherence={cohP2:.3f}")
print(f"    classical     : {accC_ft:.2f}%")
print(f"    Δ (phase − classical) = {accP_ft-accC_ft:+.2f}%")

print("\n"+"="*60); print(" VERDICT"); print("="*60)
dA=accP_inf-accC_inf; dB=accP_ft-accC_ft
print(f"  inference   Δ = {dA:+.2f}%   continuation Δ = {dB:+.2f}%")
if dB>1.0:  print("  => PHASE COUPLING HELPS when coherence can build (framework's cos φ_ij is real signal)")
elif dB<-1.0: print("  => phase coupling HURTS this trained brain (interference misaligned with learned bonds)")
else:       print("  => neutral even with continuation; phase needs training-time presence from birth")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile p.py\n"+HARNESS,"harness"),cell("!python p.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_phase_resume.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
