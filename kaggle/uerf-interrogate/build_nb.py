"""Builds uerf_interrogate.ipynb — ask the v5 checkpoint questions, then test it."""
import json, os

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

BODY = r'''
import os, sys, glob, time
import numpy as np
import torch

# ── locate the mounted Kaggle checkpoint (the v5 from-scratch run output) ─────
def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True), key=len)
    if not hits: raise FileNotFoundError(name)
    return hits[0]
BRAIN = find("uerf_brain.py"); CKPT = find("main.pt")
print("[mount] brain:", BRAIN, "\n[mount] ckpt :", CKPT, flush=True)
sys.path.insert(0, os.path.dirname(BRAIN))
import importlib.util
spec = importlib.util.spec_from_file_location("uerf_brain", BRAIN)
U = importlib.util.module_from_spec(spec); sys.modules["uerf_brain"]=U; spec.loader.exec_module(U)

dev = "cuda"; assert torch.cuda.is_available()
field = U.UERFField.load_checkpoint(CKPT, device=dev)
field.neuro_enabled = False
print(f"[brain] n_classes={field.n_classes} t_start={field.t_start} "
      f"experience={field.experience_count} bonds={int(field.C_mask.sum())}", flush=True)

# ── data + the SAME zero-mean encoding the brain was trained with ────────────
from torchvision import datasets, transforms
to_t = transforms.ToTensor()
train = datasets.MNIST("./data", train=True,  download=True)
test  = datasets.MNIST("./data", train=False, download=True)
train_imgs = torch.stack([to_t(img).view(-1) for img,_ in train])
mean_img = train_imgs.mean(0)            # frozen dataset mean image (zero-mean encoding)
proj = U.SensoryProjection(784, field.t_start, seed=42, mean_raw=mean_img).to(dev)
print(f"[encode] zero-mean projection, ||mean||={mean_img.norm():.3f}", flush=True)

N_TASK = 10
def ask(img):
    x = proj(to_t(img).view(-1).to(dev))
    pred, scores = field.eval_predict(x, return_scores=True, domain_lo=0, domain_hi=N_TASK)
    s = scores[:N_TASK]
    p = torch.softmax(s - s.max(), dim=0)        # readable confidence
    return int(pred), p.detach().cpu().numpy()

# ════════════════════════════════════════════════════════════════════════════
# PART 1 — ASK IT: one example per digit, what does it answer & how sure?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print(" ASKING THE BRAIN — one clean example of each digit 0-9")
print("="*64)
by_digit = {}
for img, lbl in test:
    if lbl not in by_digit:
        by_digit[lbl] = img
    if len(by_digit) == 10: break
for d in range(10):
    pred, p = ask(by_digit[d])
    top = p.argsort()[::-1][:3]
    conf = "  ".join(f"{i}:{p[i]*100:4.1f}%" for i in top)
    mark = "OK " if pred == d else "XX "
    print(f"  shown a '{d}'  ->  says '{pred}'  {mark} | top: {conf}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# PART 2 — TEST IT: accuracy + confusion + noise robustness
# ════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import classification_report, confusion_matrix
N = 2500
idx = np.random.RandomState(1).permutation(len(test))[:N]
preds, tgts = [], []
t0 = time.time()
for k, i in enumerate(idx):
    img, lbl = test[int(i)]
    pr, _ = ask(img)
    preds.append(pr); tgts.append(lbl)
    if (k+1) % 500 == 0:
        print(f"  [test] {k+1}/{N}  {(k+1)/(time.time()-t0):.1f}/s", flush=True)
preds, tgts = np.array(preds), np.array(tgts)
print("\n" + "="*64); print(f" TEST — {N} samples, full 10-way"); print("="*64)
print(classification_report(tgts, preds, digits=3, zero_division=0))
print("Confusion (rows=true 0-9):"); print(confusion_matrix(tgts, preds))
print(f"Accuracy: {100*np.mean(preds==tgts):.2f}%", flush=True)

# noise robustness
def ask_noisy(img, f):
    t = to_t(img).view(-1)
    t = (t + torch.randn_like(t)*f).clamp(0,1)
    x = proj(t.to(dev))
    pred = field.eval_predict(x, domain_lo=0, domain_hi=N_TASK)
    return int(pred)
for f in (0.25, 0.5):
    c = t_ = 0
    for i in idx[:1000]:
        img, lbl = test[int(i)]
        c += int(ask_noisy(img, f) == lbl); t_ += 1
    print(f"  noise σ={f}: {100*c/t_:.1f}%", flush=True)

r = field.report()
print(f"\n[brain] state_dist: {r['state_dist']}")
print("[done]", flush=True)
'''

def cell(src, cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "id":cid,"source":src.splitlines(keepends=True)}
nb = {"cells":[cell(PIN,"pin"), cell(BODY,"body")],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.12"}},
      "nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "uerf_interrogate.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out)
