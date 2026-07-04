"""Builds uerf_valence.ipynb — the framework's OWN readout (Eq 6 Valence) vs
magnitude, on GPU. Eq 6: V(s)=α<s,R> - λ||s-I||². The -λ||s-I||² term penalizes
a slot that's loud but deviates from its identity pattern — the spurious cross-
domain activations that read as 'forgetting'. All readouts computed from ONE
relaxation per image (capture the settled teach-field, score every way).
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

HARNESS = r'''"""VALENCE (Eq 6) vs MAGNITUDE readout — framework-native, on GPU."""
import numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms
to_t=transforms.ToTensor(); dev="cuda"
mn_tr=datasets.MNIST("./d",train=True,download=True); mn_te=datasets.MNIST("./d",train=False,download=True)
tr=[i for i,_ in mn_tr]; trl=torch.tensor([l for _,l in mn_tr])
by={d:(trl==d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
Xte=torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte=torch.tensor([l for _,l in mn_te])
mean_img=torch.stack([to_t(i).view(-1) for i in tr[:8000]]).mean(0)
proj=SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev)   # stable frame
enc=lambda pil: proj(to_t(pil).view(-1).to(dev))
PROBE=torch.randperm(len(Xte))[:800].tolist()

torch.manual_seed(369)
f=UERFField(n_max=200,n_initial=120,d=32,input_dim=64,n_classes=15); f.start_emergent()
label={}
def teach(cs,shots=8):
    for r in range(shots):
        for d in cs:
            x=enc(tr[by[d][r]])
            if d not in label:
                sig=(x.unsqueeze(-1)*f.c[:f.input_dim]).sum(0); label[d]=f.birth_category(sig)
            tv=torch.zeros(f.n_classes,device=dev); tv[label[d]]=1.0
            f.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=f.n_classes)
teach(range(0,5)); teach(range(5,10))
inv={v:d for d,v in label.items()}; lo,hi=min(label.values()),max(label.values())+1
t0,t1=f.t_start,f.t_end

READOUTS=[("magnitude",None),
          ("valence α1 λ0.5",(1.0,0.5)),("valence α1 λ1",(1.0,1.0)),
          ("valence α1 λ2",(1.0,2.0)),("valence α1 λ4",(1.0,4.0)),
          ("valence α0.5 λ2",(0.5,2.0))]

def enc_te(i): return proj(Xte[i].to(dev))

def score(name,cfg,s,c):
    if cfg is None: return s.norm(dim=-1)                     # magnitude
    a,l=cfg
    res=(s*c).sum(-1); dev=((s-c)**2).sum(-1)
    return a*res - l*dev                                      # Eq 6 valence

def run(classes_eval, slo, shi):
    # one relaxation per image; score every readout from the captured field
    cnt={n:0 for n,_ in READOUTS}; tot=0
    holder={}
    def cap():
        holder['s']=f.s[t0:t1].clone(); return f.s[t0:t1].norm(dim=-1)
    for i in PROBE:
        y=int(Yte[i])
        if classes_eval is not None and y not in classes_eval: continue
        f.eval_predict(enc_te(i),n_relax=8,return_scores=True,predict_fn=cap)
        s=holder['s']; c=f.c[t0:t1]
        tot+=1
        for name,cfg in READOUTS:
            sc=score(name,cfg,s,c).clone()
            mask=torch.zeros_like(sc,dtype=torch.bool); mask[slo:shi]=True
            sc=sc.masked_fill(~mask,-1e30)
            cnt[name]+= (inv.get(int(sc.argmax()))==y)
    return {n:100*cnt[n]/max(tot,1) for n,_ in READOUTS}, tot

print("="*66); print(" VALENCE (Eq 6) vs MAGNITUDE — framework-native readout"); print("="*66)
allr,_=run(set(range(10)),lo,hi)
# 10-way (all classes) and memory-frame (0-4 among 0-4)
loA=min(label[d] for d in range(5)); hiA=max(label[d] for d in range(5))+1
mem,_=run(set(range(5)),loA,hiA)          # 0-4 memory in its own frame
full,_=run(set(range(5)),lo,hi)           # 0-4 judged among all 10
print(f"\n  {'readout':20}{'10-way':>9}{'0-4|0-4':>9}{'0-4|0-9':>9}")
for name,_ in READOUTS:
    print(f"  {name:20}{allr[name]:>8.1f}{mem[name]:>9.1f}{full[name]:>9.1f}",flush=True)
base=allr['magnitude']; best=max((allr[n],n) for n,_ in READOUTS)
print(f"\n  magnitude 10-way {base:.1f}%  ->  best {best[1]} {best[0]:.1f}%  (Δ {best[0]-base:+.1f})")
if best[0]-base>2: print("  => the framework's OWN readout (Eq 6) recovers conserved signal magnitude misses")
else: print("  => valence ~ magnitude here; conserved signal lives elsewhere")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile v.py\n"+HARNESS,"harness"),cell("!python v.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_valence.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
