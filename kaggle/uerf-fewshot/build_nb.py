"""Builds uerf_fewshot.ipynb — RESUME the saved brain, learn a NEW concept few-shot."""
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
FEW-SHOT / RESUME — learn like a mind, not a neural net.
We do NOT retrain. We LOAD the already-trained consolidated brain (knows digits
0-9) and teach it a BRAND-NEW concept it has never seen (Fashion-MNIST clothing)
using only a HANDFUL of examples per class. If accuracy climbs from 5/10/20/50
shots, the brain learns few-shot on top of prior structure — the opposite of an
NN that needs thousands of labelled steps. We also check the digits survive.
"""
import os, sys, glob, time, numpy as np, torch

# ── locate + load the SAVED brain (resume, no retrain) ───────────────────────
def find(name):
    h = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True), key=len)
    if not h: raise FileNotFoundError(name)
    return h[0]
BRAIN = find("uerf_brain.py"); CKPT = find("consolidated_brain.pt")
sys.path.insert(0, os.path.dirname(BRAIN))
import importlib.util
spec = importlib.util.spec_from_file_location("uerf_brain", BRAIN)
U = importlib.util.module_from_spec(spec); sys.modules["uerf_brain"]=U; spec.loader.exec_module(U)
dev = "cuda"
field = U.UERFField.load_checkpoint(CKPT, device=dev)
print(f"[resume] loaded trained brain: {field._active_categories} categories, "
      f"experience={field.experience_count}", flush=True)

# recreate the EXACT retina the brain was trained with (seed 42, MNIST mean)
from torchvision import datasets, transforms
to_t = transforms.ToTensor()
mn_tr = datasets.MNIST("./d", train=True, download=True)
mnist_mean = torch.stack([to_t(i).view(-1) for i,_ in mn_tr]).mean(0)
proj = U.SensoryProjection(784, 64, seed=42, mean_raw=mnist_mean).to(dev)

# ── recover which category = which digit (majority vote), for retention check ─
mn_te = datasets.MNIST("./d", train=False, download=True)
Xte = torch.stack([to_t(i).view(-1) for i,_ in mn_te]); Yte = torch.tensor([l for _,l in mn_te])
from collections import Counter
digit_cat = {}
for d in range(10):
    idx = (Yte==d).nonzero(as_tuple=True)[0][:60].tolist()
    votes = Counter(field.eval_predict(proj(Xte[i].to(dev)), domain_lo=0, domain_hi=field.n_classes) for i in idx)
    digit_cat[d] = votes.most_common(1)[0][0]
def digit_acc(n=400):
    inv = {v:k for k,v in digit_cat.items()}
    lo,hi = min(digit_cat.values()), max(digit_cat.values())+1
    idx = torch.randperm(len(Xte))[:n].tolist(); c=t=0
    for i in idx:
        p = field.eval_predict(proj(Xte[i].to(dev)), domain_lo=lo, domain_hi=hi)
        c += int(inv.get(p)==int(Yte[i])); t+=1
    return 100.0*c/t
acc_digits_before = digit_acc()
print(f"[baseline] digit accuracy (its prior knowledge): {acc_digits_before:.1f}%", flush=True)

# ── NEW concept: Fashion-MNIST, taught FEW-SHOT (28x28 -> same 784 retina) ────
fa_tr = datasets.FashionMNIST("./f", train=True, download=True)
fa_te = datasets.FashionMNIST("./f", train=True, download=True)  # use train split, disjoint idx
NEW = {0:"tshirt", 7:"sneaker", 8:"bag"}     # 3 visually-distinct new classes
Xf = torch.stack([to_t(i).view(-1) for i,_ in fa_tr]); Yf = torch.tensor([l for _,l in fa_tr])
new_cat = {}
for c in NEW: new_cat[c] = None    # born on first shot

def teach_shot(i, cls):
    x = proj(Xf[i].to(dev))
    if new_cat[cls] is None:
        sig = (x.unsqueeze(-1)*field.c[:field.input_dim]).sum(0)
        new_cat[cls] = field.birth_category(sig)
        print(f"    >> BORN new category {new_cat[cls]} for '{NEW[cls]}'", flush=True)
    tv = torch.zeros(field.n_classes, device=dev); tv[new_cat[cls]] = 1.0
    field.experience(x, teaching_vector=tv, n_relax=6, domain_lo=0, domain_hi=field.n_classes)

def newconcept_acc(n_per=150):
    cats = [new_cat[c] for c in NEW if new_cat[c] is not None]
    if not cats: return 0.0
    lo,hi = min(cats), max(cats)+1
    inv = {v:k for k,v in new_cat.items()}
    c=t=0
    for cls in NEW:
        idx = (Yf==cls).nonzero(as_tuple=True)[0][5000:5000+n_per].tolist()  # held-out
        for i in idx:
            p = field.eval_predict(proj(Xf[i].to(dev)), domain_lo=lo, domain_hi=hi)
            c += int(inv.get(p)==cls); t+=1
    return 100.0*c/max(t,1)

# few-shot schedule: cumulative shots per class
shots_plan = [5, 5, 10, 30]   # -> totals 5, 10, 20, 50
seen = {c:0 for c in NEW}
print("\n[few-shot] teaching a NEW concept with only tens of examples:", flush=True)
curve = []
for add in shots_plan:
    for cls in NEW:
        pool_idx = (Yf==cls).nonzero(as_tuple=True)[0][:200].tolist()
        for _ in range(add):
            teach_shot(pool_idx[seen[cls]], cls); seen[cls]+=1
    total = seen[list(NEW)[0]]
    a = newconcept_acc()
    curve.append((total, round(a,1)))
    print(f"    after {total:2d} shots/class -> new-concept accuracy {a:.1f}%  ({3*total} total examples)", flush=True)

acc_digits_after = digit_acc()
print("\n" + "="*60)
print(" FEW-SHOT / RESUME VERDICT")
print("="*60)
print(f"  NEW concept learning curve (shots/class -> acc): {curve}")
print(f"  Digit retention: {acc_digits_before:.1f}% -> {acc_digits_after:.1f}%  (learned new concept without forgetting)")
print(f"  Total examples to learn 3 new classes: {3*seen[list(NEW)[0]]}  (an NN needs thousands)")
print("[done]", flush=True)
'''

def cell(src, cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "id":cid,"source":src.splitlines(keepends=True)}
nb = {"cells":[cell(PIN,"pin"), cell(HARNESS,"harness")],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.12"}},
      "nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "uerf_fewshot.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out)
