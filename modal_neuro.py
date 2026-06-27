"""
neuro_validate.py — Validate full neuromodulator system on Phase 3 checkpoint.

Phase A (identity baseline): load Phase 3 ckpt, eval MNIST with neuro_enabled=False.
Phase B (neuro enabled):     load Phase 3 ckpt, run 1000 MNIST steps, log neuro traces,
                              re-eval accuracy and compare.

Raw pixel input — no pretrained encoder, no ML calibration. Fixed random projection only.

Run: python neuro_validate.py <hf_token>
Requires: GPU, uerf_brain.py and uerf_lifetime.py in sys.path or same dir.
"""
import os, sys, json, math, shutil
import torch
from huggingface_hub import hf_hub_download

HF_DATASET = "BlackLoks/uerf-checkpoints"
N_SENSORY   = 128   # matches CONFIG['n_sensory'] in uerf_lifetime


def load_mnist_hf():
    """Load MNIST via HuggingFace datasets — avoids dead torchvision URLs."""
    from datasets import load_dataset
    ds = load_dataset("ylecun/mnist")
    def to_tensor_pairs(split):
        pairs = []
        for row in split:
            img = torch.tensor(list(row['image'].get_flattened_data()), dtype=torch.float32) / 255.0
            pairs.append((img, int(row['label'])))
        return pairs
    tr = to_tensor_pairs(ds['train'])
    te = to_tensor_pairs(ds['test'])
    return tr, te


def run_neuro(hf_token: str, neuro_enabled: bool, n_train_steps: int = 0) -> dict:
    ws = "/tmp/uerf_neuro"
    os.makedirs(ws, exist_ok=True)
    if ws not in sys.path:
        sys.path.insert(0, ws)

    def grab(fn, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename=fn, local_dir=ws, token=hf_token)
        if rename:
            t = os.path.join(ws, rename)
            shutil.copy(p, t)
        return os.path.join(ws, rename or fn)

    grab("neuro_brain.py", rename="uerf_brain.py")
    grab("phase3_main.pt")

    import importlib
    U = importlib.import_module("uerf_brain")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(ws, "phase3_main.pt")
    field = U.UERFField.load_checkpoint(ckpt, device=dev)
    field.neuro_enabled = neuro_enabled

    print(f"\n[neuro={neuro_enabled}] checkpoint loaded — "
          f"neuro_enabled={field.neuro_enabled}, "
          f"initial levels={field._neuro}", flush=True)

    # Raw pixel projection — no pretrained encoder, no statistical calibration.
    # MNIST images are 1×28×28 = 784 floats, fixed random project to N_SENSORY.
    proj = U.SensoryProjection(784, N_SENSORY, seed=42)
    proj.to(dev)

    print("[data] loading MNIST via HuggingFace datasets ...", flush=True)
    tr, te = load_mnist_hf()
    print(f"[data] {len(tr)} train / {len(te)} test", flush=True)

    def eval_mnist(n=1000, tag=""):
        correct_raw = 0
        for i in range(n):
            img, lbl = te[i]
            x = proj(img.view(-1).to(dev))
            x = x / x.norm().clamp(min=1e-6)
            pred, scores = field.eval_predict(x, n_relax=8, return_scores=True)
            # domain-restrict to MNIST slots [0:10]
            if scores is not None:
                domain_scores = scores[0:10]
                if int(domain_scores.argmax()) == lbl:
                    correct_raw += 1
        acc = 100.0 * correct_raw / n
        print(f"[{tag}] MNIST eval {n} samples: {acc:.1f}%", flush=True)
        return acc

    print("\n── Phase A: baseline eval (before any training) ──", flush=True)
    acc_before = eval_mnist(1000, "before")

    neuro_trace = []
    da_when_wrong, da_when_right = [], []
    if n_train_steps > 0:
        print(f"\n── Training: {n_train_steps} MNIST steps (neuro={neuro_enabled}) ──",
              flush=True)
        # tr already loaded above
        for step in range(n_train_steps):
            idx = step % len(tr)
            img, lbl = tr[idx]
            x = proj(img.view(-1).to(dev))
            x = x / x.norm().clamp(min=1e-6)
            tv = torch.zeros(field.n_classes, device=dev)
            tv[lbl] = 1.0
            # Domain-restrict the closed-loop DA check to MNIST slots [0:10] so
            # foreign CIFAR/TI slots can't flip the prediction to 'wrong' and
            # pollute the wrong/right DA buckets (must match the eval below).
            field.experience(x, teaching_vector=tv, n_relax=6,
                             domain_lo=0, domain_hi=10)
            # Closed-loop check: did the brain predict right BEFORE teaching,
            # and what did DA do in response? (set inside experience())
            pc = getattr(field, "_prediction_correct", 0.5)
            da = field._neuro["da"]
            if pc >= 0.5:
                da_when_right.append(da)
            else:
                da_when_wrong.append(da)
            if (step + 1) % 200 == 0:
                nm = field._neuro.copy()
                print(f"  step {step+1:4d}: ACh={nm['ach']:.3f} DA={nm['da']:.3f} "
                      f"NA={nm['na']:.3f} Sero={nm['sero']:.3f}  "
                      f"(last pred {'RIGHT' if pc >= 0.5 else 'WRONG'})", flush=True)
                neuro_trace.append({'step': step + 1, **{k: round(v, 4) for k, v in nm.items()}})

        print("\n── Post-training eval ──", flush=True)
        acc_after = eval_mnist(1000, "after")
    else:
        acc_after = acc_before

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    da_wrong_mean = _mean(da_when_wrong)
    da_right_mean = _mean(da_when_right)
    print(f"\n[neuro={neuro_enabled}] CLOSED-LOOP DA CHECK:", flush=True)
    print(f"  steps WRONG: {len(da_when_wrong):4d}  mean DA = {da_wrong_mean}", flush=True)
    print(f"  steps RIGHT: {len(da_when_right):4d}  mean DA = {da_right_mean}", flush=True)
    if da_wrong_mean is not None and da_right_mean is not None:
        print(f"  ΔDA (wrong − right) = {round(da_wrong_mean - da_right_mean, 4)} "
              f"(>0 ⇒ loop fires: learns harder on misses)", flush=True)

    return {
        "neuro_enabled": neuro_enabled,
        "n_train_steps": n_train_steps,
        "acc_before": round(acc_before, 1),
        "acc_after":  round(acc_after,  1),
        "neuro_levels_final": {k: round(v, 4) for k, v in field._neuro.items()},
        "neuro_trace": neuro_trace,
        "da_when_wrong_mean": da_wrong_mean,
        "da_when_right_mean": da_right_mean,
        "n_wrong": len(da_when_wrong),
        "n_right": len(da_when_right),
    }


def main(token):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    brain_path = os.path.join(os.path.dirname(__file__), "uerf_brain.py")
    print(f"[main] uploading neuro_brain.py to HF ({os.path.getsize(brain_path)//1024}KB)...",
          flush=True)
    api.upload_file(
        path_or_fileobj=brain_path,
        path_in_repo="neuro_brain.py",
        repo_id=HF_DATASET,
        repo_type="dataset",
    )
    print("[main] upload done", flush=True)

    print("[main] running neuro=OFF baseline ...", flush=True)
    r_off   = run_neuro(token, False, 0)
    print("[main] running neuro=ON + 1000 train steps ...", flush=True)
    r_on_tr = run_neuro(token, True,  1000)

    print("\n\n" + "=" * 65)
    print("NEUROMODULATION REPORT — Phase 3 checkpoint / MNIST")
    print("=" * 65)
    print(f"{'':28} {'neuro=OFF':>14} {'neuro=ON+train':>14}")
    print("-" * 58)
    print(f"{'MNIST acc (before training)':28} {r_off['acc_before']:>14.1f} {r_on_tr['acc_before']:>14.1f}")
    print(f"{'MNIST acc (after 1000 steps)':28} {'—':>14} {r_on_tr['acc_after']:>14.1f}")
    print()
    print("Neuro levels at end of training run:")
    for k, v in r_on_tr['neuro_levels_final'].items():
        print(f"  {k:6s}: {v:.4f}")
    print()
    print("Neuro trace (every 200 steps):")
    for row in r_on_tr['neuro_trace']:
        print(f"  step {row['step']:4d}  ACh={row['ach']:.4f}  DA={row['da']:.4f}  "
              f"NA={row['na']:.4f}  Sero={row['sero']:.4f}")

    print()
    print("CLOSED-LOOP DA (prediction error → plasticity):")
    dw, dr = r_on_tr.get('da_when_wrong_mean'), r_on_tr.get('da_when_right_mean')
    print(f"  mean DA when WRONG ({r_on_tr.get('n_wrong')} steps): {dw}")
    print(f"  mean DA when RIGHT ({r_on_tr.get('n_right')} steps): {dr}")
    if dw is not None and dr is not None:
        print(f"  ΔDA (wrong − right) = {round(dw - dr, 4)}  "
              f"(>0 ⇒ brain learns harder from its mistakes)")

    results = {"neuro_off": r_off, "neuro_on_train": r_on_tr}
    out = os.path.join(os.path.dirname(__file__), "neuro_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main] results written to {out}", flush=True)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python neuro_validate.py <hf_token>")
        sys.exit(1)
    main(sys.argv[1])
