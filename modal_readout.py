"""
modal_readout.py — run eval_readout.py (domain-masked vs unmasked readout) on
the Phase 3 checkpoint across all three domains, on Modal GPU. Importable module
with NO top-level app.run(); drive from a separate runner.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-readout", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run_readout(domain: str, ckpt_filename: str, hf_token: str) -> dict:
    import os, sys, json, subprocess, shutil
    from huggingface_hub import hf_hub_download
    ws = f"/tmp/uerf_readout_{domain}"
    os.makedirs(ws, exist_ok=True)
    # Phase 3 ran with phase2_brain.py / phase2_lifetime.py — use those for compat
    def grab(fn, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=fn,
                            local_dir=ws, token=hf_token)
        if rename:
            t = os.path.join(ws, rename); shutil.copy(p, t); return t
        return p
    grab("phase2_brain.py",    rename="uerf_brain.py")
    grab("phase2_lifetime.py", rename="uerf_lifetime.py")
    grab("eval_readout.py")
    grab(ckpt_filename)
    ckpt = os.path.join(ws, ckpt_filename)
    out = os.path.join(ws, f"readout_{domain}.json")
    cmd = [sys.executable, os.path.join(ws, "eval_readout.py"),
           "--ckpt", ckpt, "--name", "phase3", "--domain", domain,
           "--n_eval", "2000", "--n_calib", "300", "--n_relax", "8",
           "--seed", "42", "--out", out]
    r = subprocess.run(cmd, cwd=ws)
    if r.returncode == 0 and os.path.exists(out):
        return json.load(open(out))
    return {"error": f"returncode={r.returncode}", "domain": domain}


def main(token):
    import json
    domains = ["mnist", "cifar", "ti"]
    args = [(d, "phase3_main.pt", token) for d in domains]
    results = {}
    for (d, _, _), res in zip(args, run_readout.starmap(args)):
        results[d] = res
        print(f"\n{'='*64}\n{d}:\n{json.dumps(res, indent=2)}", flush=True)
    print("\n\n" + "=" * 64)
    print("DOMAIN-MASKED vs UNMASKED READOUT — Phase 3 checkpoint")
    print("=" * 64)
    print(f"{'domain':>7} | {'chance':>6} || {'UNMASKED (argmax over all 310 slots)':^40}")
    print(f"{'':>7} | {'':>6} || {'raw':>7} {'dir':>7} {'calib':>7} {'dir+cal':>8}  ||  "
          f"{'raw':>7} {'dir':>7} {'calib':>7} {'dir+cal':>8}  (DOMAIN-MASKED)")
    print("-" * 110)
    for d, r in results.items():
        if "error" in r:
            print(f"{d}: ERROR {r['error']}"); continue
        u, m = r['unmasked'], r['domain_masked']
        print(f"{d:>7} | {r['chance']:>6} || "
              f"{u['raw']:>7} {u['dir']:>7} {u['calib']:>7} {u['dir_calib']:>8}  ||  "
              f"{m['raw']:>7} {m['dir']:>7} {m['calib']:>7} {m['dir_calib']:>8}")
    return results
