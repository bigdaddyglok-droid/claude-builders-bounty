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
                 "huggingface_hub", "pillow")
)
app = modal.App("uerf-whitened-readout", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run(hf_token: str, ckpt: str = "phase3_main.pt", n_eval: int = 2000,
        domain=(0, 10)) -> dict:
    import os, sys, shutil
    import torch, numpy as np
    from huggingface_hub import hf_hub_download
    ws = "/tmp/ws"; os.makedirs(ws, exist_ok=True); sys.path.insert(0, ws)

    def grab(fn, rn=None):
        p = hf_hub_download(HF_DATASET, fn, repo_type="dataset", token=hf_token, local_dir=ws)
        if rn: shutil.copy(p, os.path.join(ws, rn)); return os.path.join(ws, rn)
        return p
    grab("phase2_brain.py", "uerf_brain.py")
    grab("phase2_lifetime.py", "uerf_lifetime.py")
    grab(ckpt)
    import uerf_brain as U, uerf_lifetime as L

    dev = "cuda"
    field = U.UERFField.load_checkpoint(os.path.join(ws, ckpt), device=dev)
    field._replay_disabled = True
    LO, HI = domain
    out = {"ckpt": ckpt, "domain": [LO, HI],
           "n_max": field.n_max, "input_dim": int(field.input_dim),
           "n_classes": field.n_classes, "exp": int(field.experience_count)}

    # ── 1. Recover the brain's OWN stored class means mu_k (input_dim-dim) ──────
    # _teach_bond_sum[k,j]/count[k] = outer(c_k, mu_kj * c_j)
    # => mu_kj = c_k^T (bond) c_j / (||c_k||^2 ||c_j||^2)
    t0 = field.input_dim
    sens_idx = torch.arange(field.input_dim, device=dev)
    c_sens = field.c[sens_idx]                       # (D, d)
    c_sens_n2 = (c_sens * c_sens).sum(-1).clamp(min=1e-8)   # (D,)
    bsum = field._teach_bond_sum                     # (n_classes, n_max, d, d)
    bcnt = field._teach_bond_count.clamp(min=1.0)    # (n_classes,)

    mu = torch.zeros(HI - LO, field.input_dim, device=dev)
    for ki, k in enumerate(range(LO, HI)):
        c_k = field.c[t0 + k]                        # (d,)
        ck_n2 = (c_k * c_k).sum().clamp(min=1e-8)
        bond = bsum[k, sens_idx] / bcnt[k]           # (D, d, d) = outer(c_k, mu_kj c_j)
        # c_k^T bond c_j  for every sensor j, vectorized
        left = torch.einsum('d,jde->je', c_k, bond)  # (D, d) = mu_kj * (c_k.c_k) * c_j
        proj = (left * c_sens).sum(-1)               # (D,) = mu_kj ||c_k||^2 ||c_j||^2
        mu[ki] = proj / (ck_n2 * c_sens_n2)
    out["classes_with_memory"] = int((bcnt[LO:HI] > 0).sum())

    # ── 2. Build sensory feature vectors x for the eval set (brain's input) ────
    encoder = L.UniversalEncoder(device=dev)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=42); proj.to(dev)
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
    return out


def main(ckpts=None):
    import os, json
    tok = os.environ["HF_READ_TOKEN"]
    ckpts = ckpts or ["phase3_main.pt"]
    allres = {}
    for ck in ckpts:
        print(f"\n{'='*60}\nWHITENED READOUT — {ck}\n{'='*60}", flush=True)
        r = run.remote(tok, ckpt=ck)
        print(json.dumps(r, indent=2), flush=True)
        allres[ck] = r
    return allres
