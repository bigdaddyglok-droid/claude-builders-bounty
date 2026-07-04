"""Builds uerf_transfer.ipynb — FORWARD TRANSFER test on the real UERF API."""
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
FORWARD TRANSFER — does prior knowledge make NEW learning faster?
Seq brain: learns A (digits 0-4), THEN B (5-9), tracking B's learning curve.
Control brain: learns B (5-9) COLD, tracking the same curve.
If Seq's B-curve rises faster than Control's -> positive forward transfer
(the brain reused structure it learned in A). Also checks A retention.
Uses the REAL UERF API: experience(x, teaching_vector=...) + eval_predict.
"""
import os, time, numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms

dev = "cuda"; torch.manual_seed(369)
to_t = transforms.ToTensor()
tr = datasets.MNIST("./d", train=True,  download=True)
te = datasets.MNIST("./d", train=False, download=True)
Xtr = torch.stack([to_t(i).view(-1) for i,_ in tr]); Ytr = torch.tensor([l for _,l in tr])
Xte = torch.stack([to_t(i).view(-1) for i,_ in te]); Yte = torch.tensor([l for _,l in te])
mean_img = Xtr.mean(0)                                  # frozen zero-mean encoding
proj = SensoryProjection(784, 64, seed=42, mean_raw=mean_img).to(dev)

def pool(Y, classes):
    m = torch.zeros(len(Y), dtype=torch.bool)
    for c in classes: m |= (Y == c)
    return m.nonzero(as_tuple=True)[0].tolist()

STEPS = 1500
CKPT  = 250

def new_brain():
    f = UERFField(n_max=400, n_initial=200, d=32, input_dim=64, n_classes=15)
    f.start_emergent()
    f.neuro_enabled = False  # DA loop off for the transfer test (proven separately; halves cost)
    return f

def accuracy(field, label_map, classes, n=300):
    """Derive this split's live category indices from label_map, domain-restrict,
    and score. Robust to categories born mid-training."""
    cats = [label_map[c] for c in classes if c in label_map]
    if not cats:
        return 0.0
    lo, hi = min(cats), max(cats)+1
    inv = {v:k for k,v in label_map.items()}
    idx = pool(Yte, classes); rng = np.random.RandomState(2); rng.shuffle(idx)
    correct=tot=0
    for i in idx:
        if tot>=n: break
        cat = field.eval_predict(proj(Xte[i].to(dev)), domain_lo=lo, domain_hi=hi)
        correct += int(inv.get(cat)==int(Yte[i])); tot+=1
    return 100.0*correct/max(tot,1)

def train(field, label_map, Xi, Yi, idxpool, n_steps, curve_classes=None, tag=""):
    """Train n_steps on samples from idxpool; if curve_classes given, eval that
    split every CKPT steps and record the learning curve."""
    rng = np.random.RandomState(7)
    curve = []
    t0 = time.time()
    for step in range(n_steps):
        i = idxpool[rng.randint(0, len(idxpool))]
        y = int(Yi[i]); x = proj(Xi[i].to(dev))
        if y not in label_map:
            sig = (x.unsqueeze(-1) * field.c[:field.input_dim]).sum(0)
            label_map[y] = field.birth_category(sig)
        cat = label_map[y]
        tv = torch.zeros(field.n_classes, device=dev); tv[cat] = 1.0
        field.experience(x, teaching_vector=tv, n_relax=6, domain_lo=0, domain_hi=field.n_classes)
        if curve_classes is not None and (step+1) % CKPT == 0:
            a = accuracy(field, label_map, curve_classes, n=300)
            curve.append(round(a,1))
            print(f"    [{tag}] step {step+1:4d} {(step+1)/(time.time()-t0):.1f}/s  acc={a:.1f}%", flush=True)
    return curve

A_cls, B_cls = [0,1,2,3,4], [5,6,7,8,9]
poolA, poolB = pool(Ytr, A_cls), pool(Ytr, B_cls)

# ── SEQ brain: learn A, then B (tracking B curve) ────────────────────────────
print("\n[SEQ] learning Phase A (0-4) ...", flush=True)
seq = new_brain(); seq_map = {}
train(seq, seq_map, Xtr, Ytr, poolA, STEPS, tag="seqA")
accA_pre = accuracy(seq, seq_map, A_cls)
print(f"[SEQ] Phase A learned: {accA_pre:.1f}%   now transferring to B ...", flush=True)
seq_curve = train(seq, seq_map, Xtr, Ytr, poolB, STEPS, curve_classes=B_cls, tag="seqB")
accA_post = accuracy(seq, seq_map, A_cls)
accB_seq  = accuracy(seq, seq_map, B_cls)

# ── CONTROL brain: learn B cold (tracking B curve) ──────────────────────────
print("\n[CONTROL] learning Phase B (5-9) COLD ...", flush=True)
ctl = new_brain(); ctl_map = {}
ctl_curve = train(ctl, ctl_map, Xtr, Ytr, poolB, STEPS, curve_classes=B_cls, tag="ctlB")
accB_ctl = accuracy(ctl, ctl_map, B_cls)

print("\n" + "="*60)
print(" FORWARD TRANSFER VERDICT")
print("="*60)
print(f"  A retention (Seq): {accA_pre:.1f}% -> {accA_post:.1f}%")
print(f"  B learning curve (every {CKPT} steps):")
print(f"    SEQ (knew A first): {seq_curve}")
print(f"    CONTROL (cold)    : {ctl_curve}")
if seq_curve and ctl_curve:
    early = seq_curve[0] - ctl_curve[0]
    final = accB_seq - accB_ctl
    print(f"  Early advantage (first checkpoint): {early:+.1f}%")
    print(f"  Final B accuracy: Seq={accB_seq:.1f}%  Control={accB_ctl:.1f}%  (Δ {final:+.1f}%)")
    if early > 2.0:
        print("  => POSITIVE FORWARD TRANSFER: prior knowledge accelerated new learning.")
    elif early < -2.0:
        print("  => NEGATIVE transfer (interference).")
    else:
        print("  => NEUTRAL: no measurable forward transfer (retention still holds).")
print("[done]", flush=True)
'''

def cell(src, cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "id":cid,"source":src.splitlines(keepends=True)}
nb = {"cells":[cell(PIN,"pin"),
               cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
               cell("%%writefile transfer.py\n"+HARNESS,"harness"),
               cell("!python transfer.py 2>&1","run")],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.12"}},
      "nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "uerf_transfer.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out, "brain bytes", len(BRAIN))
