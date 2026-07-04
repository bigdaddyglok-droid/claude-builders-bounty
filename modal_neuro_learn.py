"""
modal_neuro_learn.py — Does neuromodulation help the brain LEARN MORE on a hard
domain (CIFAR-100), and does it cost retention on a mastered domain (MNIST)?

Clean A/B from the SAME Phase-3 checkpoint:
  A: neuro_enabled=False, train N steps on CIFAR-100 (offset 10)
  B: neuro_enabled=True,  train N steps on CIFAR-100  — identical data order (seeded)

After each, whitened readout on:
  - CIFAR-100  [10,110)  → did it learn more class structure?
  - MNIST      [0,10)    → did neuro-driven plasticity erode old memory?

If neuro helps:  B_cifar_whitened > A_cifar_whitened, with B_mnist ≈ A_mnist.
The neuro channels (ACh novelty, DA RPE) should actually MOVE here because
CIFAR is not mastered — unlike the flat MNIST-only validation run.

Drive: with app.run(): main(token)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "huggingface_hub", "pillow")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
app = modal.App("uerf-neuro-learn", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"

# task -> (domain_lo, domain_hi, loader_name, label_offset)
TASKS = {
    "mnist": (0, 10, "load_mnist", 0),
    "cifar": (10, 110, "load_cifar100", 10),
    "ti":    (110, 310, "load_tiny_imagenet", 110),
}


def _whitened_readout(field, U, L, task, n_eval, dev):
    """Recover the brain's stored class means, run native/recon/whitened readout
    on `task`'s domain. Returns a metrics dict. (Lifted from modal_whitened_readout.)"""
    import torch, numpy as np
    LO, HI, loader_name, offset = TASKS[task]
    HI = min(HI, field.n_classes)

    t0 = field.input_dim
    sens_idx = torch.arange(field.input_dim, device=dev)
    c_sens = field.c[sens_idx]
    c_sens_n2 = (c_sens * c_sens).sum(-1).clamp(min=1e-8)
    bsum = getattr(field, "_teach_bond_sum", None)
    bcnt = getattr(field, "_teach_bond_count", None)
    use_acc = bsum is not None and bcnt is not None

    mu = torch.zeros(HI - LO, field.input_dim, device=dev)
    for ki, k in enumerate(range(LO, HI)):
        c_k = field.c[t0 + k]
        ck_n2 = (c_k * c_k).sum().clamp(min=1e-8)
        if use_acc and float(bcnt[k]) > 0:
            bond = bsum[k, sens_idx] / bcnt[k].clamp(min=1.0)
        else:
            bond = field.C[t0 + k, sens_idx]
        left = torch.einsum('d,jde->je', c_k, bond)
        projd = (left * c_sens).sum(-1)
        mu[ki] = projd / (ck_n2 * c_sens_n2)
    classes_with_memory = int((mu.norm(dim=-1) > 1e-8).sum())

    encoder = L.UniversalEncoder(device=dev)
    n_sens = L.CONFIG['n_sensory']
    proj = U.SensoryProjection(512, n_sens, seed=42); proj.to(dev)
    _, te = getattr(L, loader_name)()
    g = torch.Generator().manual_seed(0)
    idxs = torch.randperm(len(te), generator=g)[:min(n_eval, len(te))].tolist()
    X, ys = [], []
    for i in idxs:
        img, lbl = te[i]
        x = proj(encoder.encode(img.unsqueeze(0))[0])
        x = x / x.norm().clamp(min=1e-6)
        X.append(x); ys.append(int(lbl) + offset)
    X = torch.stack(X, 0)
    ys = np.array(ys)

    def acc(pred): return round(100.0 * (np.array(pred) == ys).mean(), 2)

    # native masked readout — runs full brain dynamics per sample, so cap it
    n_native = min(len(ys), 600)
    pm = []
    for i in range(n_native):
        holder = {}; orig = field.predict
        def _cap(_o=orig):
            r = _o(); holder['s'] = r.clone() if r is not None else None; return r
        field.predict = _cap
        field.eval_predict(X[i], n_relax=8)
        field.predict = orig
        sc = holder.get('s')
        pm.append(int(sc[LO:HI].argmax()) + LO if sc is not None else -1)
    native = round(100.0 * (np.array(pm) == ys[:n_native]).mean(), 2)

    # reconstructed matched filter
    recon = acc((X @ mu.T).pow(2).argmax(1).cpu().numpy() + LO)

    # whitened Mahalanobis (ridge sweep)
    Xc = X - X.mean(0, keepdim=True)
    Sigma = (Xc.T @ Xc) / max(1, len(ys) - 1)
    D = field.input_dim
    eye = torch.eye(D, device=dev)
    whit = {}
    for ridge in (1e-3, 1e-2, 1e-1, 5e-1):
        lam = ridge * torch.diagonal(Sigma).mean()
        Winv = torch.linalg.inv(Sigma + lam * eye)
        Wmu = mu @ Winv
        score_w = X @ Wmu.T - 0.5 * (Wmu * mu).sum(-1).unsqueeze(0)
        whit[f"ridge_{ridge:g}"] = acc(score_w.argmax(1).cpu().numpy() + LO)
    best_whit = max(whit.values())

    import gc
    del X, Xc, Sigma, mu, Winv
    gc.collect(); torch.cuda.empty_cache()
    return {"domain": [LO, HI], "classes_with_memory": classes_with_memory,
            "native_masked": native, "reconstructed": recon,
            "whitened": whit, "best_whitened": best_whit}


@app.function(gpu="A100-40GB", timeout=7200, memory=80000, cpu=8.0)
def train_and_readout(hf_token: str, neuro_enabled: bool, n_train: int,
                      train_task: str = "cifar", seed: int = 1234,
                      n_eval: int = 2000) -> dict:
    import os, sys, shutil, time
    import torch
    from huggingface_hub import hf_hub_download

    ws = "/tmp/uerf_nl"; os.makedirs(ws, exist_ok=True); sys.path.insert(0, ws)

    def grab(fn, rn=None):
        p = hf_hub_download(HF_DATASET, fn, repo_type="dataset",
                            token=hf_token, local_dir=ws)
        if rn:
            shutil.copy(p, os.path.join(ws, rn)); return os.path.join(ws, rn)
        return p

    grab("neuro_brain.py", "uerf_brain.py")        # the NEW neuro-enabled brain
    grab("phase2_lifetime.py", "uerf_lifetime.py")
    ckpt = grab("phase3_main.pt")

    import uerf_brain as U, uerf_lifetime as L
    dev = "cuda"
    field = U.UERFField.load_checkpoint(ckpt, device=dev)
    field.neuro_enabled = neuro_enabled
    # reset neuro to identity so both arms start equal
    field._neuro = {'ach': 1.0, 'da': 1.0, 'na': 1.0, 'sero': 1.0}
    field._mean_valence_prev = 0.0
    field._input_surprise = 1.0

    LO, HI, loader_name, offset = TASKS[train_task]
    encoder = L.UniversalEncoder(device=dev)
    n_sens = L.CONFIG['n_sensory']
    proj = U.SensoryProjection(512, n_sens, seed=42); proj.to(dev)
    tr, _ = getattr(L, loader_name)()

    print(f"\n[neuro={neuro_enabled}] training {n_train} steps on {train_task} "
          f"(classes {LO}-{HI-1}), seed={seed}", flush=True)

    # deterministic data order — both arms see the EXACT same sequence
    torch.manual_seed(seed)
    if hasattr(tr, 'targets'):
        all_lbls = tr.targets.tolist() if torch.is_tensor(tr.targets) else list(tr.targets)
    elif hasattr(tr, '_labels'):
        all_lbls = list(tr._labels)
    else:
        all_lbls = [tr[i][1] for i in range(len(tr))]
    order = torch.randint(0, len(all_lbls), (n_train,), generator=None).tolist()

    teach = torch.zeros(field.n_classes, device=dev)
    trace = []
    bsz = 8
    t0 = time.time()
    for s0 in range(0, n_train, bsz):
        chunk = order[s0:s0 + bsz]
        imgs = torch.stack([tr[j][0] for j in chunk])
        feats = encoder.encode(imgs)
        for i, j in enumerate(chunk):
            x = proj(feats[i])
            x = x / x.norm().clamp(min=1e-6)
            teach.zero_(); teach[all_lbls[j] + offset] = 1.0
            field.experience(x, teach, n_relax=L.CONFIG['n_relax'])
        step = s0 + bsz
        if step % 500 < bsz:
            nm = field._neuro
            print(f"  step {step:5d}  {step/(time.time()-t0):5.1f} ex/s  "
                  f"ACh={nm['ach']:.3f} DA={nm['da']:.3f} NA={nm['na']:.3f} "
                  f"Sero={nm['sero']:.3f}", flush=True)
            trace.append({'step': step, **{k: round(v, 4) for k, v in nm.items()}})

    final_neuro = {k: round(v, 4) for k, v in field._neuro.items()}

    # readouts: target domain (learning) + MNIST (retention)
    print(f"[neuro={neuro_enabled}] readout {train_task}...", flush=True)
    r_target = _whitened_readout(field, U, L, train_task, n_eval, dev)
    print(f"[neuro={neuro_enabled}] readout mnist (retention)...", flush=True)
    r_mnist = _whitened_readout(field, U, L, "mnist", n_eval, dev)

    import gc; del field; gc.collect(); torch.cuda.empty_cache()
    return {
        "neuro_enabled": neuro_enabled, "n_train": n_train,
        "train_task": train_task, "seed": seed,
        "final_neuro": final_neuro, "neuro_trace": trace,
        "target": {train_task: r_target},
        "retention": {"mnist": r_mnist},
    }


def main(token, n_train=5000, train_task="cifar"):
    import os, json
    from huggingface_hub import HfApi
    import concurrent.futures

    api = HfApi(token=token)
    brain_path = os.path.join(os.path.dirname(__file__), "uerf_brain.py")
    print(f"[main] uploading neuro_brain.py ({os.path.getsize(brain_path)//1024}KB)...",
          flush=True)
    api.upload_file(path_or_fileobj=brain_path, path_in_repo="neuro_brain.py",
                    repo_id=HF_DATASET, repo_type="dataset")
    print("[main] upload done", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(train_and_readout.remote, token, False, n_train, train_task, 1234)
        fb = ex.submit(train_and_readout.remote, token, True,  n_train, train_task, 1234)
        A = fa.result(); B = fb.result()

    print("\n\n" + "=" * 72, flush=True)
    print(f"NEURO LEARNING A/B — Phase3 + {n_train} steps {train_task.upper()}", flush=True)
    print("=" * 72, flush=True)
    tt = train_task
    print(f"{'metric':32} {'neuro=OFF':>12} {'neuro=ON':>12} {'Δ':>8}", flush=True)
    print("-" * 68, flush=True)
    def row(name, a, b):
        print(f"{name:32} {a:>12.2f} {b:>12.2f} {b-a:>+8.2f}", flush=True)
    row(f"{tt} native (masked)",   A['target'][tt]['native_masked'], B['target'][tt]['native_masked'])
    row(f"{tt} reconstructed",     A['target'][tt]['reconstructed'],  B['target'][tt]['reconstructed'])
    row(f"{tt} WHITENED (best)",   A['target'][tt]['best_whitened'],  B['target'][tt]['best_whitened'])
    print("-" * 68, flush=True)
    row("mnist retention (whiten)", A['retention']['mnist']['best_whitened'],
        B['retention']['mnist']['best_whitened'])
    print(f"\nfinal neuro (ON arm): {B['final_neuro']}", flush=True)
    print("neuro trace (ON arm):", flush=True)
    for r in B['neuro_trace']:
        print(f"  step {r['step']:5d}  ACh={r['ach']:.4f}  DA={r['da']:.4f}  "
              f"NA={r['na']:.4f}  Sero={r['sero']:.4f}", flush=True)

    out = {"neuro_off": A, "neuro_on": B, "n_train": n_train, "train_task": train_task}
    p = os.path.join(os.path.dirname(__file__), "neuro_learn_results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[main] results -> {p}", flush=True)
    return out
