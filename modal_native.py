"""
modal_native.py — Native full-field readout on the Phase 3 checkpoint.

Loads phase3_main.pt and compares how the brain reads its own memory:

  raw         — default predict(): per-slot magnitude argmax  (the lossy readout)
  native_self — full settled teach-field matched against prototypes built ONLY
                from the brain's own replay memories (20 remembered MNIST/class)
  native_wide — same full-field readout, prototypes built from the brain's
                memories PLUS a wider labeled calibration pass (ceiling check)

No external classifier is trained — recognition is cosine match of the full
field response. The wide pass only supplies more reference examples.

Runner uploads the local (native-readout) uerf_brain.py to HF first.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-native", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run_native(hf_token: str, n_eval: int = 2000, n_wide: int = 1000,
               n_relax: int = 8) -> dict:
    import os, sys, json, shutil
    import torch
    import numpy as np
    from huggingface_hub import hf_hub_download

    ws = "/tmp/uerf_native"
    os.makedirs(ws, exist_ok=True)
    sys.path.insert(0, ws)

    def grab(fn, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename=fn, local_dir=ws, token=hf_token)
        if rename:
            shutil.copy(p, os.path.join(ws, rename))
        return os.path.join(ws, rename or fn)

    grab("native_brain.py",    rename="uerf_brain.py")
    grab("phase2_lifetime.py", rename="uerf_lifetime.py")
    grab("phase3_main.pt")

    import uerf_brain as U
    import uerf_lifetime as L

    dev = "cuda"
    field = U.UERFField.load_checkpoint(os.path.join(ws, "phase3_main.pt"), device=dev)
    field._replay_disabled = True
    lo, hi = 0, 10   # MNIST slots

    n_mem = sum(len(b) for k, b in field._replay_per_class.items() if 0 <= k < 10)
    print(f"[native] loaded — exp={int(field.experience_count)} "
          f"locked={int((field._class_locked & field.alive_mask).sum())} "
          f"MNIST replay memories={n_mem}", flush=True)

    # feature pipeline (identical to training/eval)
    encoder = L.UniversalEncoder(device=dev)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=42)
    proj.to(dev)

    def feats(split, n):
        tr, te = L.load_mnist()
        ds = te if split == 'test' else tr
        xs, ys = [], []
        for i in range(min(n, len(ds))):
            img, lbl = ds[i]
            x = proj(encoder.encode(img.unsqueeze(0))[0])
            x = x / x.norm().clamp(min=1e-6)
            xs.append(x); ys.append(int(lbl))
        return xs, ys

    te_x, te_y = feats('test', n_eval)
    te_y = np.asarray(te_y)

    def acc_native(tag):
        preds = [field.eval_predict_native(x, n_relax=n_relax, domain_lo=lo, domain_hi=hi)
                 for x in te_x]
        a = 100.0 * (np.asarray(preds) == te_y).mean()
        print(f"[native:{tag}] MNIST {len(te_y)} samples = {a:.1f}%", flush=True)
        return round(a, 1)

    def acc_raw():
        preds = [field.eval_predict(x, n_relax=n_relax) for x in te_x]
        a = 100.0 * (np.asarray(preds) == te_y).mean()
        print(f"[raw] MNIST {len(te_y)} samples = {a:.1f}%", flush=True)
        return round(a, 1)

    raw = acc_raw()

    # (1) catalog from the brain's OWN memory only
    print("\n[native] building catalog from replay memory only...", flush=True)
    field.build_self_catalog(n_relax=n_relax)
    native_self = acc_native("self-memory")

    # (2) catalog widened with a labeled calibration pass
    print(f"\n[native] widening catalog with {n_wide} calibration examples...", flush=True)
    cal_x, cal_y = feats('train', n_wide)
    field.build_self_catalog(n_relax=n_relax, extra_xs=cal_x, extra_ys=cal_y)
    native_wide = acc_native("wide")

    return {
        "n_eval": int(len(te_y)),
        "mnist_replay_memories": int(n_mem),
        "raw_magnitude": raw,
        "native_self_memory": native_self,
        "native_wide_calib": native_wide,
        "probe_ceiling": 86.6,
    }


def main(token):
    import json, os
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    brain = os.path.join(os.path.dirname(__file__), "uerf_brain.py")
    print(f"[main] uploading native_brain.py ({os.path.getsize(brain)//1024}KB)...",
          flush=True)
    api.upload_file(path_or_fileobj=brain, path_in_repo="native_brain.py",
                    repo_id=HF_DATASET, repo_type="dataset")

    print("[main] running native readout eval...", flush=True)
    r = run_native.remote(token)
    print(f"\n{json.dumps(r, indent=2)}", flush=True)

    print("\n" + "=" * 56)
    print("NATIVE FULL-FIELD READOUT — Phase 3 checkpoint (MNIST)")
    print("=" * 56)
    print(f"  raw magnitude argmax (old default) : {r['raw_magnitude']:>6.1f}%")
    print(f"  native (brain's own memory)        : {r['native_self_memory']:>6.1f}%")
    print(f"  native (memory + wide calibration) : {r['native_wide_calib']:>6.1f}%")
    print(f"  supervised probe ceiling           : {r['probe_ceiling']:>6.1f}%")
    return r
