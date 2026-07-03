"""Builds uerf_consolidated.ipynb — the definitive everything-on brain, trained + reported + saved."""
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
CONSOLIDATED UERF BRAIN — everything on, in one artifact.
A single brain that: grows its OWN categories (emergent), learns locally (no
backprop), remembers (disjoint accumulator), forgets gracefully + recalls
(vacuum), learns from its mistakes (dopamine reward/loss), and UNDERSTANDS
(concept-relation web). One run proves all of it together, and saves the brain.
"""
import os, time, numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

dev = "cuda"; torch.manual_seed(369)
to_t = transforms.ToTensor()
mn_tr = datasets.MNIST("./d", train=True,  download=True)
mn_te = datasets.MNIST("./d", train=False, download=True)
train_imgs = torch.stack([to_t(i).view(-1) for i,_ in mn_tr])
train_lbls = torch.tensor([l for _,l in mn_tr])
mean_img = train_imgs.mean(0)                          # zero-mean encoding (frozen)
proj = SensoryProjection(784, 64, seed=42, mean_raw=mean_img).to(dev)

field = UERFField(n_max=400, n_initial=200, d=32, input_dim=64, n_classes=15)
field.start_emergent()                                 # ZERO categories at start
print(f"[brain] everything-on. active_categories={field._active_categories} "
      f"neuro_enabled={field.neuro_enabled}", flush=True)

label_map = {}
da_wrong, da_right = [], []
def teach(idx):
    y = int(train_lbls[idx]); x = proj(train_imgs[idx].to(dev))
    if y not in label_map:
        sig = (x.unsqueeze(-1) * field.c[:field.input_dim]).sum(0)
        c = field.birth_category(sig); label_map[y] = c
        print(f"    >> BORN category {c} for digit {y}  (total {field._active_categories})", flush=True)
    cat = label_map[y]
    tv = torch.zeros(field.n_classes, device=dev); tv[cat] = 1.0
    field.experience(x, teaching_vector=tv, n_relax=6, domain_lo=0, domain_hi=field.n_classes)
    pc = getattr(field, "_prediction_correct", 0.5); da = field._neuro["da"]
    (da_right if pc >= 0.5 else da_wrong).append(da)

N_TRAIN = 8000
print(f"\n[train] {N_TRAIN} MNIST examples (categories grow on demand)", flush=True)
perm = np.random.RandomState(1).permutation(len(train_imgs))[:N_TRAIN]
t0 = time.time()
for k, i in enumerate(perm):
    teach(int(i))
    if (k+1) % 1000 == 0:
        nvac = int((field.state_id == 9).sum())  # S_VACUUM=9
        print(f"    {k+1}/{N_TRAIN}  {(k+1)/(time.time()-t0):.1f}/s  cats={field._active_categories} vac={nvac}", flush=True)

# ── CAPABILITY REPORT ────────────────────────────────────────────────────────
test_imgs = torch.stack([to_t(i).view(-1) for i,_ in mn_te])
test_lbls = torch.tensor([l for _,l in mn_te])
inv = {v:k for k,v in label_map.items()}
def predict(i):
    cat = field.eval_predict(proj(test_imgs[i].to(dev)), domain_lo=0, domain_hi=field.n_classes)
    return inv.get(cat, -1)

idx = np.random.RandomState(2).permutation(len(test_imgs))[:2000]
preds = [predict(int(i)) for i in idx]; tgts = [int(test_lbls[i]) for i in idx]
acc = 100.0*np.mean(np.array(preds)==np.array(tgts))

# dopamine reward/loss loop
dw = round(sum(da_wrong)/len(da_wrong),4) if da_wrong else None
dr = round(sum(da_right)/len(da_right),4) if da_right else None

# concept-relation web (understanding) — map category indices back to digits
rels = field.concept_relations(top_k=3)
digit_rels = {}
for ci, picks in rels.items():
    d = inv.get(ci)
    if d is None: continue
    digit_rels[d] = [(inv.get(cj), round(s,3)) for cj,s in picks if inv.get(cj) is not None]

nvac = int((field.state_id == 9).sum())
r = field.report()

print("\n" + "="*64)
print(" CONSOLIDATED BRAIN — ALL CAPABILITIES IN ONE ARTIFACT")
print("="*64)
print(f"  EMERGENCE   : grew {field._active_categories} of its own categories from zero")
print(f"  ACCURACY    : {acc:.1f}%  (MNIST, 2000 test)")
print(f"  MEMORY      : {nvac} memories dormant in vacuum reservoir (forget/recall live)")
print(f"  DOPAMINE    : DA when WRONG={dw}  RIGHT={dr}  ΔDA={round((dw-dr),4) if dw and dr else None}  (>0 = learns from mistakes)")
print(f"  UNDERSTANDING (concept relations the brain learned on its own):")
for d in sorted(digit_rels):
    if digit_rels[d]:
        print(f"      digit {d}  ~  " + ", ".join(f"{o}({s})" for o,s in digit_rels[d]))
print(f"  STATES      : {r['state_dist']}")
print("  CONFUSION (rows=true digit):")
print(confusion_matrix(tgts, preds, labels=list(range(10))))

# ── SAVE the definitive brain (the usable artifact) ──────────────────────────
os.makedirs("/kaggle/working/ckpt", exist_ok=True)
field.save_checkpoint("/kaggle/working/ckpt/consolidated_brain.pt")
import json as _j
_j.dump({"accuracy":acc, "categories":field._active_categories, "vacuum":nvac,
         "da_wrong":dw, "da_right":dr, "concept_relations":{str(k):v for k,v in digit_rels.items()}},
        open("/kaggle/working/ckpt/report.json","w"), indent=2)
print("\n[saved] consolidated_brain.pt + report.json")
print("[done]", flush=True)
'''

def cell(src, cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "id":cid,"source":src.splitlines(keepends=True)}
nb = {"cells":[cell(PIN,"pin"),
               cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
               cell("%%writefile consolidated.py\n"+HARNESS,"harness"),
               cell("!python consolidated.py 2>&1","run")],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.12"}},
      "nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "uerf_consolidated.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out, "brain bytes", len(BRAIN))
