"""
modal_readout_native.py — native consolidated-memory readout on phase3_main.pt.
Everything runs INSIDE the Modal container (own disk + GPU). No local staging,
no local upload — avoids the host disk-full problem. Drive with:
    with app.run(): main()
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "huggingface_hub", "pillow")
)
app = modal.App("uerf-readout-native", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run_readout(hf_token: str, n_eval: int = 2000) -> dict:
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
    grab("phase3_main.pt")
    import uerf_brain as U, uerf_lifetime as L

    dev = "cuda"
    field = U.UERFField.load_checkpoint(os.path.join(ws, "phase3_main.pt"), device=dev)
    field._replay_disabled = True

    out = {}
    locked = (field._class_locked & field.alive_mask)
    n_locked = int(locked.sum())
    out["n_max"] = field.n_max; out["n_classes"] = field.n_classes
    out["experience_count"] = int(field.experience_count)
    out["n_locked"] = n_locked
    out["replay_entries"] = int(sum(len(v) for v in field._replay_per_class.values()))
    if n_locked:
        lc = field._locked_class[locked].cpu().numpy()
        uniq, cnt = np.unique(lc, return_counts=True)
        out["locked_class_min"] = int(uniq.min()); out["locked_class_max"] = int(uniq.max())
        out["distinct_locked_classes"] = int(len(uniq))
        out["mnist_locked"] = {int(c): int(n) for c, n in zip(uniq, cnt) if 0 <= c < 10}

    encoder = L.UniversalEncoder(device=dev)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=42); proj.to(dev)
    _, te = L.load_mnist(); xs, ys = [], []
    for i in range(min(n_eval, len(te))):
        img, lbl = te[i]
        x = proj(encoder.encode(img.unsqueeze(0))[0]); x = x / x.norm().clamp(min=1e-6)
        xs.append(x); ys.append(int(lbl))
    ys = np.array(ys); out["n_eval"] = len(ys)
    LO, HI = 0, 10

    def input_signature(x):
        isig = (x.unsqueeze(-1) * field.c[:field.input_dim]).sum(0)
        return isig / isig.norm().clamp(min=1e-6)

    # native consolidated-memory readout
    if n_locked:
        li = locked.nonzero(as_tuple=True)[0]
        lcls = field._locked_class[li]; c_lock = field.c[li]
        def reson(x):
            isig = input_signature(x)
            r = (c_lock * isig.unsqueeze(0)).sum(-1)
            sc = torch.full((field.n_classes,), -1e9, device=dev)
            for k in range(LO, HI):
                m = (lcls == k)
                if m.any(): sc[k] = r[m].mean()
            return sc
        pred = [int(reson(x).argmax()) for x in xs]
        out["resonance_vote"] = round(100.0 * (np.array(pred) == ys).mean(), 1)

    # raw baselines — capture teach-slot norms via a predict hook (version-proof)
    praw, pm = [], []
    for x in xs:
        holder = {}
        orig = field.predict
        def _cap(_o=orig):
            r = _o()
            holder['scores'] = r.clone() if r is not None else None
            return r
        field.predict = _cap
        full = field.eval_predict(x, n_relax=8)
        field.predict = orig
        praw.append(int(full) if not isinstance(full, tuple) else int(full[0]))
        sc = holder.get('scores')
        pm.append(int(sc[LO:HI].argmax()) + LO if sc is not None else -1)
    out["raw_full310"] = round(100.0 * (np.array(praw) == ys).mean(), 1)
    out["raw_mnist_masked"] = round(100.0 * (np.array(pm) == ys).mean(), 1)
    return out


def main():
    import os, json
    tok = os.environ["HF_READ_TOKEN"]
    r = run_readout.remote(tok)
    print("\n" + "=" * 56)
    print("NATIVE CONSOLIDATED-MEMORY READOUT — phase3_main.pt")
    print("=" * 56)
    print(json.dumps(r, indent=2))
    return r
