"""Builds uerf_benchmark.ipynb — adapts MNISTAdvancedBenchmarkSuite to the UERF field."""
import json, os

BENCH = r'''
import os, sys, json, math, time
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# UERF adaptation of MNISTAdvancedBenchmarkSuite.
#
# The original suite assumes a differentiable nn.Module CNN:
#   torch.load -> model(data) -> logits ; FGSM via loss.backward()/data.grad.
# The UERF checkpoint is a physics field (dict of tensors), evaluated one
# sample at a time via field.eval_predict(x) on a raw-pixel -> random-projection
# vector, with a relaxation loop and discrete state routing. It is NOT autograd-
# differentiable, so:
#   * preprocessing = raw [0,1] pixels (what the brain was trained on), NOT the
#     (0.1307,0.3081) CNN normalization the original used.
#   * Stage 4 FGSM is replaced by a GRADIENT-FREE L-inf worst-of-K perturbation
#     stress test (no input gradient exists). Clearly labeled as NOT FGSM.
# Stages 1-3 (statistical / geometric / stochastic) are faithful.
# ─────────────────────────────────────────────────────────────────────────────

import glob
def _find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input")
    return sorted(hits, key=len)[0]   # shallowest match
print("[mount] /kaggle/input contents:")
for p in sorted(glob.glob("/kaggle/input/*")):
    print("   ", p, flush=True)
BRAIN_PY = _find("uerf_brain.py")
CKPT     = _find("main.pt")
print(f"[mount] brain  = {BRAIN_PY}\n[mount] ckpt   = {CKPT}", flush=True)

import importlib.util
sys.path.insert(0, os.path.dirname(BRAIN_PY))
spec = importlib.util.spec_from_file_location("uerf_brain", BRAIN_PY)
U = importlib.util.module_from_spec(spec); sys.modules["uerf_brain"] = U
spec.loader.exec_module(U)

dev = "cuda" if torch.cuda.is_available() else "cpu"
assert dev == "cuda", "GPU REQUIRED — refusing to run on CPU"
print("device:", torch.cuda.get_device_name(0), flush=True)

field = U.UERFField.load_checkpoint(CKPT, device=dev)
field.neuro_enabled = False
print(f"[brain] n_max={field.n_max} d={field.d} n_classes={field.n_classes} "
      f"t_start={field.t_start} t_end={field.t_end} "
      f"experience={field.experience_count}", flush=True)

RAW_DIM   = 784
N_SENSORY = field.t_start            # sensors occupy [0:t_start]
N_TASK    = 10                       # MNIST digits 0-9
N_RELAX   = 8
proj = U.SensoryProjection(RAW_DIM, N_SENSORY, seed=42).to(dev)

def predict(img_chw01):
    """img_chw01: (1,28,28) float tensor in [0,1].  -> int prediction 0-9."""
    x = proj(img_chw01.reshape(-1).to(dev))
    pred, scores = field.eval_predict(x, n_relax=N_RELAX, return_scores=True)
    return int(scores[:N_TASK].argmax())

# ── data ─────────────────────────────────────────────────────────────────────
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

raw_test = datasets.MNIST(root="./data", train=False, download=True)   # PIL + int
to_tensor = transforms.ToTensor()                                       # -> [0,1]

def subset_indices(n, seed=0):
    g = np.random.RandomState(seed)
    idx = np.arange(len(raw_test)); g.shuffle(idx)
    return idx[:n].tolist()

def run_over(indices, make_tensor, tag):
    preds, tgts = [], []
    t0 = time.time()
    for k, i in enumerate(indices):
        img, lbl = raw_test[i]
        t = make_tensor(img)
        preds.append(predict(t)); tgts.append(int(lbl))
        if (k + 1) % 500 == 0:
            print(f"  [{tag}] {k+1}/{len(indices)}  {(k+1)/(time.time()-t0):.2f} ex/s", flush=True)
    return np.array(preds), np.array(tgts)

results = {}

# ── STAGE 1: STATISTICAL BREAKDOWN ───────────────────────────────────────────
N1 = 2500
idx1 = subset_indices(N1, seed=1)
print(f"\n=== STAGE 1: STATISTICAL BREAKDOWN  (n={N1}, full 10-way) ===", flush=True)
p1, t1 = run_over(idx1, lambda img: to_tensor(img), "stat")
print(classification_report(t1, p1, digits=4, zero_division=0))
cm = confusion_matrix(t1, p1)
print("Confusion Matrix (rows=true 0-9, cols=pred):")
print(cm)
acc1 = float(np.mean(p1 == t1))
results["accuracy"] = acc1
results["confusion_matrix"] = cm.tolist()
print(f"Overall accuracy: {acc1*100:.4f}%", flush=True)

# ── STAGE 2: GEOMETRIC INVARIANCE ────────────────────────────────────────────
ROT, TRANS = 30, 0.15
N2 = 1500
idx2 = subset_indices(N2, seed=2)
geom_tf = transforms.Compose([
    transforms.RandomRotation(degrees=(ROT, ROT)),
    transforms.RandomAffine(degrees=0, translate=(TRANS, TRANS)),
    transforms.ToTensor(),
])
print(f"\n=== STAGE 2: GEOMETRIC INVARIANCE  (rot={ROT}deg, trans={TRANS}, n={N2}) ===", flush=True)
p2, t2 = run_over(idx2, lambda img: geom_tf(img), "geom")
acc2 = float(np.mean(p2 == t2))
results["geometric_accuracy"] = acc2
print(f"Spatial-invariance accuracy: {acc2*100:.4f}%", flush=True)

# ── STAGE 3: STOCHASTIC RESILIENCE ───────────────────────────────────────────
NOISE_F, MASK_R = 0.4, 0.25
N3 = 1500
idx3 = subset_indices(N3, seed=3)
print(f"\n=== STAGE 3: STOCHASTIC RESILIENCE  (noise={NOISE_F}, mask={MASK_R}, n={N3}) ===", flush=True)

def noisy(img):
    t = to_tensor(img)
    return (t + torch.randn_like(t) * NOISE_F).clamp(0.0, 1.0)
def masked(img):
    t = to_tensor(img).clone()
    t[torch.rand_like(t) < MASK_R] = 0.0
    return t

pn, tn = run_over(idx3, noisy, "noise")
pm, tm = run_over(idx3, masked, "mask")
acc_n = float(np.mean(pn == tn)); acc_m = float(np.mean(pm == tm))
results["noise_accuracy"] = acc_n
results["mask_accuracy"] = acc_m
print(f"Additive Gaussian noise accuracy: {acc_n*100:.4f}%", flush=True)
print(f"Pixel masking ({MASK_R*100:.0f}% zeroed) accuracy: {acc_m*100:.4f}%", flush=True)

# ── STAGE 4: GRADIENT-FREE L-inf STRESS (substitute for FGSM) ────────────────
# The physics readout has no input gradient, so FGSM is inapplicable. Instead:
# for each sample the brain ALREADY classifies correctly, draw K random
# sign perturbations of magnitude epsilon in raw-pixel space; the sample is
# "robust" only if ALL K perturbations preserve the correct prediction.
# This is a black-box L-inf worst-of-K test, NOT FGSM.
EPS, K = 0.15, 6
N4 = 500
idx4 = subset_indices(N4, seed=4)
print(f"\n=== STAGE 4: GRADIENT-FREE L-inf STRESS (eps={EPS}, K={K}, NOT FGSM) ===", flush=True)
robust = 0; considered = 0; t0 = time.time()
for k, i in enumerate(idx4):
    img, lbl = raw_test[i]; lbl = int(lbl)
    base = to_tensor(img)
    if predict(base) != lbl:
        continue                      # only attack what it gets right (as original)
    considered += 1
    survived = True
    for _ in range(K):
        sign = torch.sign(torch.randn_like(base))
        adv = (base + EPS * sign).clamp(0.0, 1.0)
        if predict(adv) != lbl:
            survived = False; break
    if survived:
        robust += 1
    if (considered) % 100 == 0 and considered > 0:
        print(f"  [adv] considered={considered} robust={robust} "
              f"{considered/(time.time()-t0):.2f} clean/s", flush=True)
adv_acc = (robust / considered) if considered else 0.0
results["adversarial_robustness"] = adv_acc
results["adversarial_considered"] = considered
print(f"Clean-correct samples attacked: {considered}", flush=True)
print(f"L-inf worst-of-{K} robustness: {adv_acc*100:.4f}%", flush=True)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" UERF BENCHMARK SUMMARY — raw-pixel MNIST checkpoint")
print("=" * 60)
print(f"  Stage 1 clean accuracy        : {results['accuracy']*100:7.3f}%")
print(f"  Stage 2 geometric (rot+trans) : {results['geometric_accuracy']*100:7.3f}%")
print(f"  Stage 3 gaussian noise        : {results['noise_accuracy']*100:7.3f}%")
print(f"  Stage 3 pixel masking         : {results['mask_accuracy']*100:7.3f}%")
print(f"  Stage 4 L-inf worst-of-{K} rob. : {results['adversarial_robustness']*100:7.3f}%")
print("=" * 60)
with open("/kaggle/working/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("[done] wrote benchmark_results.json", flush=True)
'''

nb = {
    "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "id": "bench", "source": BENCH.splitlines(keepends=True)}
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(os.path.dirname(__file__), "uerf_benchmark.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
