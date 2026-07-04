"""Builds uerf_retina.ipynb — A/B: Bessel/harmonic retina vs random projection."""
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
RETINA A/B — physics-legal Bessel/harmonic front-end vs random projection.
Two brains trained identically on MNIST; one sees through HarmonicRetina (the
framework's Bessel J_h(kR)*3/6/9*fractal math, magnitude-read), the other
through the random SensoryProjection. We compare CLEAN accuracy and — the real
prize — GEOMETRIC INVARIANCE (shifted + rotated digits), where the random retina
collapsed to ~15%. Does the framework's own retina give the field eyes?
"""
import os, time, numpy as np, torch
from uerf_brain import UERFField, SensoryProjection, HarmonicRetina
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

dev="cuda"
to_t=transforms.ToTensor()
mn_tr=datasets.MNIST("./d",train=True,download=True); mn_te=datasets.MNIST("./d",train=False,download=True)
tr_imgs=[img for img,_ in mn_tr]; tr_lbl=torch.tensor([l for _,l in mn_tr])
te_imgs=[img for img,_ in mn_te]; te_lbl=torch.tensor([l for _,l in mn_te])
mean_img=torch.stack([to_t(i).view(-1) for i in tr_imgs[:8000]]).mean(0)

def flat(pil): return to_t(pil).view(-1)
def shift(pil,dx=3,dy=2): return TF.affine(pil,angle=0,translate=[dx,dy],scale=1.0,shear=0)
def rot(pil,deg=25): return TF.rotate(pil,deg)

STEPS=1500
def build(kind):
    torch.manual_seed(369)
    if kind=="bessel":
        enc=HarmonicRetina(28, mean_raw=mean_img).to(dev); nsen=enc.n_sensory
    else:
        enc=SensoryProjection(784,64,seed=42,mean_raw=mean_img).to(dev); nsen=64
    f=UERFField(n_max=400,n_initial=200,d=32,input_dim=nsen,n_classes=15)
    f.start_emergent()
    return f,enc

def run(kind):
    field,enc=build(kind); label={}
    rng=np.random.RandomState(7); t0=time.time()
    for step in range(STEPS):
        i=rng.randint(0,len(tr_imgs)); y=int(tr_lbl[i]); x=enc(flat(tr_imgs[i]))
        if y not in label:
            sig=(x.unsqueeze(-1)*field.c[:field.input_dim]).sum(0); label[y]=field.birth_category(sig)
        tv=torch.zeros(field.n_classes,device=dev); tv[label[y]]=1.0
        field.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=field.n_classes)
    inv={v:k for k,v in label.items()}; lo,hi=min(label.values()),max(label.values())+1
    def acc(tf,n=600):
        idx=np.random.RandomState(2).permutation(len(te_imgs))[:n].tolist(); c=t=0
        for i in idx:
            x=enc(flat(tf(te_imgs[i])) if tf else flat(te_imgs[i]))
            p=field.eval_predict(x,domain_lo=lo,domain_hi=hi)
            c+=int(inv.get(p)==int(te_lbl[i])); t+=1
        return 100.0*c/t
    clean=acc(None); sh=acc(shift); ro=acc(rot)
    print(f"[{kind}] n_sensory={field.input_dim} clean={clean:.1f} shifted={sh:.1f} rotated={ro:.1f} "
          f"({STEPS/(time.time()-t0):.1f}/s)",flush=True)
    return dict(clean=clean,shifted=sh,rotated=ro)

print("="*60); print(" RETINA A/B — Bessel/harmonic vs random projection"); print("="*60)
rnd=run("random")
bes=run("bessel")
print("\n"+"="*60); print(" VERDICT"); print("="*60)
print(f"  {'':10}{'RANDOM':>10}{'BESSEL':>10}{'Δ':>8}")
for k in ['clean','shifted','rotated']:
    print(f"  {k:10}{rnd[k]:>10.1f}{bes[k]:>10.1f}{bes[k]-rnd[k]:>+8.1f}")
gi_r=rnd['shifted']-rnd['clean']; gi_b=bes['shifted']-bes['clean']
print(f"\n  invariance loss (shifted-clean): random {gi_r:+.1f}  bessel {gi_b:+.1f}")
if bes['shifted']-rnd['shifted']>5: print("  => BESSEL retina gives real spatial invariance (the framework's eyes)")
else: print("  => no clear invariance win yet")
print("[done]",flush=True)
'''

def cell(src,cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,"source":src.splitlines(keepends=True)}
nb={"cells":[cell(PIN,"pin"),cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile r.py\n"+HARNESS,"harness"),cell("!python r.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_retina.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain bytes",len(BRAIN))
