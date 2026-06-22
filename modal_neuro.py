"""
modal_neuro.py — Test the neuromodulation layer on Phase 3 checkpoint.

Phase A (identity check): load Phase 3 ckpt, eval MNIST with neuro_enabled=False,
confirm accuracy matches pre-neuro baseline (52.8% calibrated / 38.8% raw).

Phase B (neuro enabled): set neuro_enabled=True, run 500 MNIST training steps,
re-eval — look for accuracy improvement vs the baseline above.

The runner uploads the local uerf_brain.py (neuro-enabled version) to HF as
"neuro_brain.py" before launching the GPU container.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-neuro", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run_neuro(hf_token: str, neuro_enabled: bool, n_train_steps: int = 0) -> dict:
    import os, sys, json, math, shutil
    import torch
    from huggingface_hub import hf_hub_download

    ws = "/tmp/uerf_neuro"
    os.makedirs(ws, exist_ok=True)
    sys.path.insert(0, ws)

    def grab(fn, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename=fn, local_dir=ws, token=hf_token)
        if rename:
            t = os.path.join(ws, rename)
            shutil.copy(p, t)
        return os.path.join(ws, rename or fn)

    # Download neuro-enabled brain (uploaded by main() before this runs)
    grab("neuro_brain.py",   rename="uerf_brain.py")
    grab("phase2_lifetime.py", rename="uerf_lifetime.py")
    grab("phase3_main.pt")
    grab("eval_full.py")

    import uerf_brain as U
    import uerf_lifetime as L

    dev = "cuda"
    ckpt = os.path.join(ws, "phase3_main.pt")
    field = U.UERFField.load_checkpoint(ckpt, device=dev)
    field._replay_disabled = True
    field.neuro_enabled = neuro_enabled

    print(f"\n[neuro={neuro_enabled}] checkpoint loaded — "
          f"neuro_enabled={field.neuro_enabled}, "
          f"levels={field._neuro}", flush=True)

    # ── Quick MNIST eval (1000 test samples) ──────────────────────────────────
    encoder = L.UniversalEncoder(device=dev)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=42)
    proj.to(dev)
    _, te = L.load_mnist()

    def eval_mnist(n=1000, tag=""):
        correct_raw, correct_calib = 0, 0
        field.calibrate_readout([
            proj(encoder.encode(te[i][0].unsqueeze(0))[0])
            for i in range(200)
        ])
        for i in range(n):
            img, lbl = te[i]
            x = proj(encoder.encode(img.unsqueeze(0))[0])
            x = x / x.norm().clamp(min=1e-6)
            scores = field.eval_predict(x, n_relax=8, return_scores=True)
            if scores is not None:
                pred_raw = int(scores.argmax())
                if pred_raw == lbl:
                    correct_raw += 1
                norms = scores.clone()
                mu = getattr(field, '_readout_mu', None)
                sd = getattr(field, '_readout_sigma', None)
                if mu is not None and sd is not None:
                    norms = (norms - mu) / sd
                if int(norms[0:10].argmax()) == lbl:
                    correct_calib += 1
        acc_raw = 100.0 * correct_raw / n
        acc_calib = 100.0 * correct_calib / n
        print(f"[{tag}] MNIST eval {n} samples: raw={acc_raw:.1f}%  "
              f"domain-calib={acc_calib:.1f}%", flush=True)
        return acc_raw, acc_calib

    print("\n── Phase A: baseline eval (before any training) ──", flush=True)
    raw_before, calib_before = eval_mnist(1000, "before")

    # ── Optional short training ────────────────────────────────────────────────
    if n_train_steps > 0 and neuro_enabled:
        print(f"\n── Phase B: {n_train_steps} MNIST training steps with neuro ON ──",
              flush=True)
        tr, _ = L.load_mnist()
        field._replay_disabled = False  # allow replay during training
        for step in range(n_train_steps):
            idx = step % len(tr)
            img, lbl = tr[idx]
            x = proj(encoder.encode(img.unsqueeze(0))[0])
            x = x / x.norm().clamp(min=1e-6)
            tv = torch.zeros(field.n_classes, device=dev)
            tv[lbl] = 1.0
            field.experience(x, teaching_vector=tv, n_relax=6)
            if (step + 1) % 100 == 0:
                nm = field._neuro
                print(f"  step {step+1}: ACh={nm['ach']:.3f} DA={nm['da']:.3f} "
                      f"NA={nm['na']:.3f} Sero={nm['sero']:.3f}", flush=True)
        field._replay_disabled = True
        print("\n── Phase B: post-training eval ──", flush=True)
        raw_after, calib_after = eval_mnist(1000, "after")
    else:
        raw_after, calib_after = raw_before, calib_before

    return {
        "neuro_enabled": neuro_enabled,
        "n_train_steps": n_train_steps,
        "raw_before": round(raw_before, 1),
        "calib_before": round(calib_before, 1),
        "raw_after": round(raw_after, 1),
        "calib_after": round(calib_after, 1),
        "neuro_levels_final": dict(field._neuro),
    }


def main(token):
    import json, os
    from huggingface_hub import HfApi

    # Upload neuro-enabled brain to HF so the GPU container can download it
    api = HfApi(token=token)
    brain_path = os.path.join(os.path.dirname(__file__), "uerf_brain.py")
    print(f"[main] uploading neuro_brain.py to HF... ({os.path.getsize(brain_path)//1024}KB)",
          flush=True)
    api.upload_file(
        path_or_fileobj=brain_path,
        path_in_repo="neuro_brain.py",
        repo_id=HF_DATASET,
        repo_type="dataset",
    )
    print("[main] upload done", flush=True)

    print("\n[main] running identity check (neuro_enabled=False)...", flush=True)
    r_off = run_neuro.remote(token, neuro_enabled=False, n_train_steps=0)
    print(f"\n[OFF] {json.dumps(r_off, indent=2)}", flush=True)

    print("\n[main] running neuro-enabled (500 MNIST steps)...", flush=True)
    r_on = run_neuro.remote(token, neuro_enabled=True, n_train_steps=500)
    print(f"\n[ON] {json.dumps(r_on, indent=2)}", flush=True)

    print("\n\n" + "=" * 60)
    print("NEUROMODULATION COMPARISON — Phase 3 checkpoint")
    print("=" * 60)
    print(f"{'':20} {'neuro=OFF':>12} {'neuro=ON':>12}")
    print("-" * 46)
    print(f"{'MNIST raw (before)':20} {r_off['raw_before']:>12.1f} {r_on['raw_before']:>12.1f}")
    print(f"{'MNIST calib (before)':20} {r_off['calib_before']:>12.1f} {r_on['calib_before']:>12.1f}")
    if r_on['n_train_steps'] > 0:
        print(f"{'MNIST raw (after)':20} {'—':>12} {r_on['raw_after']:>12.1f}")
        print(f"{'MNIST calib (after)':20} {'—':>12} {r_on['calib_after']:>12.1f}")
    print(f"{'final neuro levels':20} {'identity':>12} {str(r_on['neuro_levels_final'])}")
    return r_off, r_on
