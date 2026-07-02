"""Builds uerf_emergent.ipynb — brain grows its OWN categories: MNIST then CIFAR."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAIN = open(os.path.join(ROOT, "uerf_brain.py")).read()

PIN = '''import subprocess, sys
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu}")
if "P100" in gpu:
    print("P100 -> pinning torch 2.7.1")
    subprocess.run(["pip","install","-q","torch==2.7.1","torchvision==0.22.1"], check=True)
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
'''

HARNESS = r'''"""
EMERGENT CATEGORIES — MNIST then CIFAR-10 on ONE brain.
The brain begins with ZERO categories (start_emergent). Each time it meets a
label it has no category for, it BIRTHS one (birth_category) — born-when-needed,
tuned to the pattern that birthed it. Nothing is pre-assigned. Then we test:
did it grow 10 categories for MNIST, 10 more for CIFAR, and keep the MNIST ones?
"""
import os, time, numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms

dev = "cuda"; torch.manual_seed(369)
RAW = 32*32*3           # universal retina: 32x32x3
N_SENS = 64
# MNIST -> resize 32, gray->3ch ; CIFAR native 32x32x3
mnist_tf = transforms.Compose([transforms.Resize(32), transforms.Grayscale(3), transforms.ToTensor()])
cifar_tf = transforms.Compose([transforms.ToTensor()])
mn_tr = datasets.MNIST("./d", train=True,  download=True, transform=mnist_tf)
mn_te = datasets.MNIST("./d", train=False, download=True, transform=mnist_tf)
cf_tr = datasets.CIFAR10("./d", train=True,  download=True, transform=cifar_tf)
cf_te = datasets.CIFAR10("./d", train=False, download=True, transform=cifar_tf)

def flat(t): return t.reshape(-1)
# frozen universal mean image (zero-mean encoding) from a sample of both
samp = torch.stack([flat(mn_tr[i][0]) for i in range(2000)] +
                   [flat(cf_tr[i][0]) for i in range(2000)])
mean_img = samp.mean(0)
proj = SensoryProjection(RAW, N_SENS, seed=42, mean_raw=mean_img).to(dev)

field = UERFField(n_max=500, n_initial=140, d=32, input_dim=N_SENS, n_classes=20)
field.start_emergent()          # begin with NO categories
print(f"[brain] alive={int(field.alive_mask.sum())} active_categories={field._active_categories}", flush=True)

label_map = {}                   # (domain,label) -> category index
def teach(x_raw, glabel):
    x = proj(x_raw.to(dev))
    if glabel not in label_map:
        sens = (x.unsqueeze(-1) * field.c[:field.input_dim]).sum(0)   # d-space signature
        cat = field.birth_category(sens)
        label_map[glabel] = cat
        print(f"    >> BORN category {cat} for {glabel}  (total categories = {field._active_categories})", flush=True)
    cat = label_map[glabel]
    tv = torch.zeros(field.n_classes, device=dev); tv[cat] = 1.0
    field.experience(x, teaching_vector=tv, n_relax=6)

def stream(ds, glabel_fn, n, tag):
    print(f"\n[{tag}] streaming {n} examples", flush=True)
    perm = np.random.RandomState(1).permutation(len(ds))[:n]
    t0 = time.time()
    for k, i in enumerate(perm):
        img, lbl = ds[int(i)]
        teach(flat(img), glabel_fn(lbl))
        if (k+1) % 1000 == 0:
            print(f"    [{tag}] {k+1}/{n}  {(k+1)/(time.time()-t0):.1f}/s  cats={field._active_categories}", flush=True)

def evaluate(ds, glabel_fn, cats, n=500, tag=""):
    lo, hi = min(cats), max(cats)+1
    inv = {v:k for k,v in label_map.items()}
    perm = np.random.RandomState(2).permutation(len(ds))[:n*3]
    correct=tot=0
    for i in perm:
        if tot>=n: break
        img,lbl = ds[int(i)]
        g = glabel_fn(lbl)
        if g not in label_map: continue
        pred_cat = field.eval_predict(proj(flat(img).to(dev)), n_relax=8, domain_lo=lo, domain_hi=hi)
        correct += int(inv.get(pred_cat)==g); tot+=1
    acc = 100.0*correct/max(tot,1)
    print(f"[eval {tag}] {acc:.1f}%  ({tot} samples, cats {lo}-{hi-1})", flush=True)
    return acc

# ── PHASE 1: MNIST — brain grows 10 digit categories from nothing ────────────
stream(mn_tr, lambda l:("mnist",int(l)), 3000, "MNIST")
mnist_cats = sorted(label_map[k] for k in label_map if k[0]=="mnist")
acc_mnist_before = evaluate(mn_te, lambda l:("mnist",int(l)), mnist_cats, tag="MNIST after phase1")
print(f"[emergence] MNIST grew {len(mnist_cats)} categories", flush=True)

# ── PHASE 2: CIFAR-10 — brain grows 10 MORE categories, keeps MNIST ──────────
stream(cf_tr, lambda l:("cifar",int(l)), 3000, "CIFAR")
cifar_cats = sorted(label_map[k] for k in label_map if k[0]=="cifar")
acc_mnist_after = evaluate(mn_te, lambda l:("mnist",int(l)), mnist_cats, tag="MNIST RETENTION after CIFAR")
acc_cifar = evaluate(cf_te, lambda l:("cifar",int(l)), cifar_cats, tag="CIFAR after phase2")

print("\n" + "="*60)
print(" EMERGENT CROSS-DOMAIN RESULT")
print("="*60)
print(f"  categories grown : MNIST={len(mnist_cats)}  CIFAR={len(cifar_cats)}  total={field._active_categories}")
print(f"  MNIST acc  before CIFAR : {acc_mnist_before:.1f}%")
print(f"  MNIST acc  after  CIFAR : {acc_mnist_after:.1f}%   (retention)")
print(f"  CIFAR acc               : {acc_cifar:.1f}%")
r = field.report()
print(f"  state_dist: {r['state_dist']}")
print("[done]", flush=True)
'''

def cell(src, cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "id":cid,"source":src.splitlines(keepends=True)}
nb = {"cells":[cell(PIN,"pin"),
               cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
               cell("%%writefile emergent.py\n"+HARNESS,"harness"),
               cell("!python emergent.py 2>&1","run")],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.12"}},
      "nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "uerf_emergent.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out, "brain bytes", len(BRAIN))
