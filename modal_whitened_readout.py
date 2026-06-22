"""
modal_whitened_readout.py — does the brain's OWN memory hold the answer?

The readout the brain computes is a class-mean matched filter in sensory space:
    sensory state  s_j      = x_j * c_j           (input feature j scales c_j)
    teach bond     C[t0+k,j] = outer(c_k, mu_kj * c_j)   (mu_kj = class-k mean of x_j)
    => brain score_k  proportional to  (sum_j mu_kj * x_j)^2  =  (mu_k . x)^2

So mu_k IS the brain's stored class mean (recoverable from _teach_bond_sum), and
nearest-class-mean is bounded by inter-class mean correlation (~38% on packed
multi-domain MNIST). WHITENING the same stored means by the GLOBAL sensory
covariance (unsupervised, no labels, applied equally to every class -> cannot
cause forgetting) turns nearest-mean into Mahalanobis nearest-mean (LDA), which
decorrelates the means and should climb toward the 86% the probe proved is there.

Nothing is trained. mu_k comes from the brain. Sigma comes from unlabeled inputs.
Runs fully inside the Modal container (GPU). Drive: with app.run(): main()
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "huggingface_hub", "pillow", "kaggle")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
app = modal.App("uerf-whitened-readout", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A100-40GB", timeout=7200, memory=80000, cpu=8.0)
def run(hf_token: str, source: dict, n_eval: int = 2000,
        domain=(0, 10), kaggle_user: str = "", kaggle_key: str = "") -> dict:
    """source = {'kind':'hf','file':'phase3_main.pt'}  or
                {'kind':'kaggle','ds':'blackloko/uerf-phase1-complete-ckpt',
                 'file':'lifetime_main.pt'}"""
    import os, sys, shutil
    import torch, numpy as np
    from huggingface_hub import hf_hub_download
    ws = "/tmp/ws"; os.makedirs(ws, exist_ok=True); sys.path.insert(0, ws)

    def grab_hf(fn, rn=None):
        p = hf_hub_download(HF_DATASET, fn, repo_type="dataset", token=hf_token, local_dir=ws)
        if rn: shutil.copy(p, os.path.join(ws, rn)); return os.path.join(ws, rn)
        return p
    # loader + eval pipeline always come from the canonical HF brain
    grab_hf("phase2_brain.py", "uerf_brain.py")
    grab_hf("phase2_lifetime.py", "uerf_lifetime.py")

    label = source.get("label", source.get("file", "ckpt"))
    if source["kind"] == "hf":
        ckpt_path = grab_hf(source["file"])
    else:  # kaggle dataset
        os.environ["KAGGLE_USERNAME"] = kaggle_user
        os.environ["KAGGLE_KEY"] = kaggle_key
        # newer kaggle CLI prefers the json file with 0600 perms
        kcfg = os.path.expanduser("~/.kaggle"); os.makedirs(kcfg, exist_ok=True)
        import json as _json
        with open(os.path.join(kcfg, "kaggle.json"), "w") as _f:
            _json.dump({"username": kaggle_user, "key": kaggle_key}, _f)
        os.chmod(os.path.join(kcfg, "kaggle.json"), 0o600)
        import subprocess, zipfile
        kd = os.path.join(ws, "kag"); os.makedirs(kd, exist_ok=True)
        # download the zip (no --unzip: extract only the .pt to save disk)
        r = subprocess.run(["kaggle", "datasets", "download", "-d", source["ds"],
                            "-p", kd], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"kaggle download failed rc={r.returncode}: "
                               f"OUT={r.stdout[-800:]} ERR={r.stderr[-800:]}")
        zips = [f for f in os.listdir(kd) if f.endswith(".zip")]
        if not zips:
            raise RuntimeError(f"no zip downloaded; dir={os.listdir(kd)}")
        zp = os.path.join(kd, zips[0])
        with zipfile.ZipFile(zp) as z:
            z.extract(source["file"], kd)
        os.remove(zp)
        ckpt_path = os.path.join(kd, source["file"])

    import uerf_brain as U, uerf_lifetime as L
    dev = "cuda"
    field = U.UERFField.load_checkpoint(ckpt_path, device=dev)
    field._replay_disabled = True
    LO, HI = domain
    HI = min(HI, field.n_classes)
    out = {"label": label, "domain": [LO, HI],
           "n_max": field.n_max, "input_dim": int(field.input_dim),
           "n_classes": field.n_classes, "exp": int(field.experience_count)}
    # ── integrity audit: which switches were live when these means formed? ─────
    out["fixes"] = {k: bool(v) for k, v in getattr(field, "fixes", {}).items()}
    out["neuro_enabled"] = bool(getattr(field, "neuro_enabled", False))
    out["neuro_levels"] = {k: float(v) for k, v in getattr(field, "_neuro", {}).items()}
    rpc = getattr(field, "_replay_per_class", {}) or {}
    out["replay_entries"] = int(sum(len(v) for v in rpc.values()))
    out["replay_disabled_for_eval"] = True  # we force-disable below; never read replay

    # ── 1. Recover the brain's OWN stored class means mu_k (input_dim-dim) ──────
    # Sensor->teach bond  C[t0+k, j] = outer(c_k, mu_kj * c_j)   (mu_kj = class-k
    # mean of input feature j). Prefer the _teach_bond_sum accumulator when it is
    # present; otherwise read the finished bond straight out of C (always there,
    # even for early checkpoints that predate the accumulator).
    #   mu_kj = c_k^T (bond) c_j / (||c_k||^2 ||c_j||^2)
    t0 = field.input_dim
    sens_idx = torch.arange(field.input_dim, device=dev)
    c_sens = field.c[sens_idx]                       # (D, d)
    c_sens_n2 = (c_sens * c_sens).sum(-1).clamp(min=1e-8)   # (D,)
    bsum = getattr(field, "_teach_bond_sum", None)
    bcnt = getattr(field, "_teach_bond_count", None)
    use_acc = bsum is not None and bcnt is not None
    out["mean_source"] = "_teach_bond_sum" if use_acc else "C_matrix"

    mu = torch.zeros(HI - LO, field.input_dim, device=dev)
    for ki, k in enumerate(range(LO, HI)):
        c_k = field.c[t0 + k]                        # (d,)
        ck_n2 = (c_k * c_k).sum().clamp(min=1e-8)
        if use_acc and float(bcnt[k]) > 0:
            bond = bsum[k, sens_idx] / bcnt[k].clamp(min=1.0)
        else:
            bond = field.C[t0 + k, sens_idx]         # (D, d, d) = outer(c_k, mu_kj c_j)
        left = torch.einsum('d,jde->je', c_k, bond)  # (D, d) = mu_kj * (c_k.c_k) * c_j
        proj = (left * c_sens).sum(-1)               # (D,) = mu_kj ||c_k||^2 ||c_j||^2
        mu[ki] = proj / (ck_n2 * c_sens_n2)
    nonzero = (mu.norm(dim=-1) > 1e-8)
    out["classes_with_memory"] = int(nonzero.sum())

    # ── 2. Build sensory feature vectors x for the eval set (brain's input) ────
    encoder = L.UniversalEncoder(device=dev)
    n_sens = L.CONFIG['n_sensory']
    out["n_sensory_cfg"] = n_sens
    out["input_dim_matches_cfg"] = (n_sens == field.input_dim)
    proj = U.SensoryProjection(512, n_sens, seed=42); proj.to(dev)
    _, te = L.load_mnist()
    X, ys = [], []
    for i in range(min(n_eval, len(te))):
        img, lbl = te[i]
        x = proj(encoder.encode(img.unsqueeze(0))[0])
        x = x / x.norm().clamp(min=1e-6)
        X.append(x); ys.append(int(lbl))
    X = torch.stack(X, 0)                             # (N, D)
    ys = np.array(ys); out["n_eval"] = len(ys)

    def acc(pred): return round(100.0 * (np.array(pred) == ys).mean(), 2)

    # ── 3. Brain's actual readout (ground-truth baseline) via predict hook ─────
    praw, pm = [], []
    for i in range(len(ys)):
        holder = {}; orig = field.predict
        def _cap(_o=orig):
            r = _o(); holder['s'] = r.clone() if r is not None else None; return r
        field.predict = _cap
        full = field.eval_predict(X[i], n_relax=8)
        field.predict = orig
        praw.append(int(full) if not isinstance(full, tuple) else int(full[0]))
        sc = holder.get('s')
        pm.append(int(sc[LO:HI].argmax()) + LO if sc is not None else -1)
    out["brain_predict_full310"] = acc(praw)
    out["brain_predict_masked"]  = acc(pm)

    # ── 4. Reconstructed matched filter (mu_k . x)^2  — validates recovery ─────
    score_raw = (X @ mu.T) ** 2                       # (N, K)
    pred_raw = (score_raw.argmax(1).cpu().numpy() + LO)
    out["reconstructed_matched_filter"] = acc(pred_raw)

    # ── 5. WHITENED Mahalanobis nearest-mean (the test) ────────────────────────
    # Sigma = global sensory covariance over UNLABELED eval inputs (+ ridge).
    Xc = X - X.mean(0, keepdim=True)
    Sigma = (Xc.T @ Xc) / max(1, len(ys) - 1)         # (D, D)
    D = field.input_dim
    eye = torch.eye(D, device=dev)
    results = {}
    for ridge in (1e-3, 1e-2, 1e-1, 5e-1):
        lam = ridge * torch.diagonal(Sigma).mean()
        Winv = torch.linalg.inv(Sigma + lam * eye)    # Sigma^{-1}
        # LDA discriminant: mu_k^T W x - 0.5 mu_k^T W mu_k
        Wmu = mu @ Winv                               # (K, D)
        lin = X @ Wmu.T                               # (N, K)
        norm = 0.5 * (Wmu * mu).sum(-1)               # (K,)
        score_w = lin - norm.unsqueeze(0)
        pred_w = (score_w.argmax(1).cpu().numpy() + LO)
        results[f"ridge_{ridge:g}"] = acc(pred_w)
    out["whitened_mahalanobis"] = results
    out["best_whitened"] = max(results.values())
    # release GPU memory so a reused warm container doesn't accumulate/fragment
    import gc
    del field, X, Xc, Sigma, Winv, mu
    gc.collect(); torch.cuda.empty_cache()
    return out


# Every checkpoint: HF (lifetime) + Kaggle (Phase-1 MNIST snapshots)
SOURCES = [
    # Kaggle — Phase-1 MNIST-only snapshots (the control: separated means)
    {"kind": "kaggle", "ds": "blackloko/uerf-phase1-ckpt-step6000",
     "file": "lifetime_main.pt", "label": "kaggle/phase1-step6000"},
    {"kind": "kaggle", "ds": "blackloko/uerf-phase1-emergent-ckpt-step4000",
     "file": "lifetime_main.pt", "label": "kaggle/phase1-emergent-step4000"},
    {"kind": "kaggle", "ds": "blackloko/uerf-phase1-emergent-ckpt-step8000",
     "file": "lifetime_main.pt", "label": "kaggle/phase1-emergent-step8000"},
    {"kind": "kaggle", "ds": "blackloko/uerf-phase1-emergent-complete-ckpt",
     "file": "lifetime_main.pt", "label": "kaggle/phase1-emergent-complete"},
    {"kind": "kaggle", "ds": "blackloko/uerf-phase1-complete-ckpt",
     "file": "lifetime_main.pt", "label": "kaggle/phase1-complete"},
    # HF — lifetime gauntlet
    {"kind": "hf", "file": "emergent_main.pt", "label": "hf/emergent_main"},
    {"kind": "hf", "file": "emv2_main.pt", "label": "hf/emv2_main"},
    {"kind": "hf", "file": "phase2_main.pt", "label": "hf/phase2_main"},
    {"kind": "hf", "file": "phase3_progress.pt", "label": "hf/phase3_progress"},
    {"kind": "hf", "file": "pl1done_main.pt", "label": "hf/pl1done_main"},
    {"kind": "hf", "file": "phase3_main.pt", "label": "hf/phase3_main"},
]


def main(sources=None):
    import os, json
    tok = os.environ["HF_READ_TOKEN"]
    ku = os.environ.get("KAGGLE_USERNAME", "")
    kk = os.environ.get("KAGGLE_KEY", "")
    sources = sources or SOURCES
    allres = {}
    for src in sources:
        lab = src.get("label", src.get("file"))
        print(f"\n{'='*60}\nWHITENED READOUT — {lab}\n{'='*60}", flush=True)
        try:
            r = run.remote(tok, src, kaggle_user=ku, kaggle_key=kk)
        except Exception as e:
            r = {"label": lab, "error": str(e)[:500]}
        print(json.dumps(r, indent=2), flush=True)
        allres[lab] = r
    print("\n" + "#"*60 + "\nSUMMARY\n" + "#"*60, flush=True)
    print(f"{'checkpoint':30} {'native':>6} {'recon':>6} {'whiten':>6} "
          f"{'replay':>6} {'neuro':>6} fixes_on", flush=True)
    for lab, r in allres.items():
        if "error" in r:
            print(f"{lab:30} ERROR: {r['error'][:60]}", flush=True); continue
        on = [k for k, v in r.get("fixes", {}).items() if v]
        print(f"{lab:30} {r.get('brain_predict_masked',0):6} "
              f"{r.get('reconstructed_matched_filter',0):6} "
              f"{r.get('best_whitened',0):6} "
              f"{r.get('replay_entries',0):6} "
              f"{('ON' if r.get('neuro_enabled') else 'off'):>6} "
              f"{','.join(on) if on else '-'}", flush=True)
    with open("whitened_readout_all.json", "w") as f:
        json.dump(allres, f, indent=2)
    print("\n[saved] whitened_readout_all.json", flush=True)
    return allres
