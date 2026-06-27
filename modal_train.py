"""
modal_train.py — From-scratch split-MNIST on Modal A100, with a RECALL TEST.

Trains the corrected UERF brain (no toggles, zero-mean encoding, domain-
restricted eval) on raw-pixel MNIST:
  Phase A: digits 0-4  -> consolidate -> eval
  Phase B: digits 5-9  -> eval A retention + B

Then demonstrates the vacuum forget/recall memory model:
  - how many interior memories were demoted to the VACUUM reservoir,
  - how many Phase-A consolidated nodes were RELEASED to vacuum during B
    (graceful forgetting), and
  - a RECALL TEST: for Phase-A cues vs Phase-B cues, how many dormant (vacuum)
    nodes resonate strongly enough to be recalled — the "suddenly recalled when
    the cue returns" signal — and the retention accuracy with recall on vs off.

Run:  HF_TOKEN=... modal run modal_train.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy", "huggingface_hub")
)
app = modal.App("uerf-train", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A100", timeout=14400, memory=32768, cpu=8.0)
def run_train(hf_token: str, phase_a_steps: int = 10000, phase_b_steps: int = 10000,
              eval_n: int = 1000) -> dict:
    import os, sys, time, shutil
    import torch
    import numpy as np
    from huggingface_hub import hf_hub_download

    ws = "/tmp/uerf_train"
    os.makedirs(ws, exist_ok=True)
    sys.path.insert(0, ws)
    p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                        filename="train_brain.py", local_dir=ws, token=hf_token)
    shutil.copy(p, os.path.join(ws, "uerf_brain.py"))
    import uerf_brain as U

    dev = "cuda"
    torch.manual_seed(369)

    # ── data (raw [0,1] pixels) ───────────────────────────────────────────────
    import torchvision
    from torchvision import transforms
    t = transforms.Compose([transforms.ToTensor()])
    tr = torchvision.datasets.MNIST(ws, train=True,  download=True, transform=t)
    te = torchvision.datasets.MNIST(ws, train=False, download=True, transform=t)
    train_imgs = torch.stack([s[0].view(-1) for s in tr]).to(dev)
    train_lbls = torch.tensor([s[1] for s in tr]).to(dev)
    test_imgs  = torch.stack([s[0].view(-1) for s in te]).to(dev)
    test_lbls  = torch.tensor([s[1] for s in te]).to(dev)
    print(f"[data] train {tuple(train_imgs.shape)} test {tuple(test_imgs.shape)}", flush=True)

    # zero-mean sensory encoding (frozen dataset mean image)
    mean_img = train_imgs.mean(0)
    proj = U.SensoryProjection(784, 64, seed=42, mean_raw=mean_img).to(dev)

    field = U.UERFField(n_max=400, n_initial=200, d=32, input_dim=64, n_classes=10)
    print(f"[brain] alive={int(field.alive_mask.sum())} bonds={int(field.C_mask.sum())}", flush=True)

    def train_phase(classes, n_steps, label):
        allowed = set(classes)
        mask = torch.zeros(len(train_lbls), dtype=torch.bool, device=dev)
        for c in allowed:
            mask |= (train_lbls == c)
        pool = mask.nonzero(as_tuple=True)[0]
        rng = np.random.RandomState(field.experience_count + 1)
        t0 = time.time()
        for step in range(n_steps):
            i = pool[rng.randint(0, len(pool))].item()
            y = int(train_lbls[i])
            x = proj(train_imgs[i])
            tv = torch.zeros(field.n_classes, device=dev)
            tv[y] = 1.0
            field.experience(x, teaching_vector=tv, n_relax=6, learn=True)
            if (step + 1) % 1000 == 0:
                r = field.report()
                nvac = int((field.state_id == U.S_VACUUM).sum())
                print(f"  [{label}] {step+1}/{n_steps}  {(step+1)/(time.time()-t0):.1f} ex/s "
                      f"bonds={r['bonds']} locked={int(field._class_locked.sum())} vac={nvac}", flush=True)

    def evaluate(classes, domain, n=eval_n):
        lo, hi = domain
        allowed = set(classes)
        perm = torch.randperm(len(test_imgs))[:n * 6]
        correct = total = 0
        for idx in perm.tolist():
            if total >= n:
                break
            y = int(test_lbls[idx])
            if y not in allowed:
                continue
            x = proj(test_imgs[idx])
            pred = field.eval_predict(x, domain_lo=lo, domain_hi=hi)
            correct += int(pred == y); total += 1
        return 100.0 * correct / max(total, 1)

    def recall_events(classes, n=300):
        """Avg # of dormant (vacuum) nodes whose identity resonates with a cue of
        these classes strongly enough to be recalled. Pure measurement — mirrors
        the recall rule in _dynamics_step without mutating the brain."""
        vac = (field.state_id == U.S_VACUUM)
        if int(vac.sum()) == 0:
            return 0.0, 0
        allowed = set(classes)
        perm = torch.randperm(len(test_imgs))[:n * 6]
        counts, seen = [], 0
        for idx in perm.tolist():
            if seen >= n:
                break
            if int(test_lbls[idx]) not in allowed:
                continue
            x = proj(test_imgs[idx])
            sens = x.unsqueeze(-1) * field.c[:field.input_dim]   # (n_sensory, d)
            isig = sens.sum(0)
            isig = isig / isig.norm().clamp(min=1e-6)
            res = (field.c * isig.unsqueeze(0)).sum(-1)
            counts.append(int((vac & (res > U.RECALL_RESONANCE)).sum()))
            seen += 1
        return float(np.mean(counts)) if counts else 0.0, int(vac.sum())

    # ── Phase A ───────────────────────────────────────────────────────────────
    train_phase([0, 1, 2, 3, 4], phase_a_steps, "Phase A")
    n_lock = sum(field.consolidate_class(c, n_lock=4) for c in [0, 1, 2, 3, 4])
    locked_after_a = int(field._class_locked.sum())
    print(f"[consolidate] locked {n_lock} (total {locked_after_a})", flush=True)
    accA_within_before = evaluate([0,1,2,3,4], (0,5))
    accA_full_before   = evaluate([0,1,2,3,4], (0,10))

    # ── Phase B ───────────────────────────────────────────────────────────────
    train_phase([5, 6, 7, 8, 9], phase_b_steps, "Phase B")
    locked_after_b = int(field._class_locked.sum())
    released = locked_after_a - locked_after_b   # A-memories that faded to vacuum

    accA_within_after = evaluate([0,1,2,3,4], (0,5))
    accA_full_after   = evaluate([0,1,2,3,4], (0,10))
    accB_within       = evaluate([5,6,7,8,9], (5,10))
    acc_all           = evaluate(list(range(10)), (0,10))

    # ── Recall test ───────────────────────────────────────────────────────────
    rec_A, nvac = recall_events([0,1,2,3,4])
    rec_B, _    = recall_events([5,6,7,8,9])
    # retention accuracy with recall ON vs OFF (disable by raising the threshold)
    saved = U.RECALL_RESONANCE
    U.RECALL_RESONANCE = 2.0
    accA_within_norecall = evaluate([0,1,2,3,4], (0,5))
    U.RECALL_RESONANCE = saved

    r = field.report()
    out = {
        "accA_within_before": round(accA_within_before, 1),
        "accA_within_after":  round(accA_within_after, 1),
        "accA_full_before":   round(accA_full_before, 1),
        "accA_full_after":    round(accA_full_after, 1),
        "accB_within":        round(accB_within, 1),
        "acc_all":            round(acc_all, 1),
        "locked_after_a":     locked_after_a,
        "locked_after_b":     locked_after_b,
        "released_to_vacuum": released,
        "vacuum_nodes":       nvac,
        "recall_events_A_cue": round(rec_A, 2),
        "recall_events_B_cue": round(rec_B, 2),
        "accA_within_recall_on":  round(accA_within_after, 1),
        "accA_within_recall_off": round(accA_within_norecall, 1),
        "state_dist": r["state_dist"],
        "bonds": r["bonds"], "births": r["births"], "deaths": r["deaths"],
    }
    print("\n" + "=" * 64)
    print(" SPLIT-MNIST + VACUUM FORGET/RECALL — RESULTS")
    print("=" * 64)
    for k, v in out.items():
        if k != "state_dist":
            print(f"  {k:24}: {v}")
    print(f"  state_dist: {out['state_dist']}")
    return out


@app.local_entrypoint()
def main():
    import os, json
    from huggingface_hub import HfApi
    token = os.environ["HF_TOKEN"]
    brain = os.path.join(os.path.dirname(__file__), "uerf_brain.py")
    print(f"[main] uploading train_brain.py ({os.path.getsize(brain)//1024}KB)...", flush=True)
    HfApi(token=token).upload_file(path_or_fileobj=brain, path_in_repo="train_brain.py",
                                   repo_id=HF_DATASET, repo_type="dataset")
    print("[main] launching A100 run...", flush=True)
    r = run_train.remote(token)
    print("\n" + json.dumps(r, indent=2))
