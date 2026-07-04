"""
modal_eval.py — Run UERF checkpoint eval on Modal GPU.

Downloads checkpoints + code from BlackLoks/uerf-checkpoints (HF dataset),
runs eval_checkpoint.py on each consolidated checkpoint, prints results.

Usage (from this repo directory):
    python modal_eval.py

Requires:
    - ~/.modal/credentials.toml with valid token
    - HF_TOKEN env var or hardcoded below (write token to access private dataset)
"""
import ssl, os

# SSL patch for Anthropic egress proxy (only needed in this container)
_orig_ssl = ssl.create_default_context
def _patched_ssl(*args, **kwargs):
    ctx = _orig_ssl(*args, **kwargs)
    try:
        ctx.load_verify_locations('/tmp/anthropic_root_ca.pem')
    except Exception:
        pass
    return ctx
ssl.create_default_context = _patched_ssl

import modal

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy",
        "scikit-learn",
        "huggingface_hub",
        "pillow",
    )
)

app = modal.App("uerf-eval", image=image)

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set via env, never hardcode
HF_DATASET = "BlackLoks/uerf-checkpoints"

# ── Eval function ─────────────────────────────────────────────────────────────
@app.function(
    gpu="A10G",           # 24 GB VRAM — handles the 6.9 GB emv2 checkpoint
    timeout=7200,         # 2 hours max
    memory=32768,         # 32 GB RAM
)
def run_eval(ckpt_name: str, ckpt_filename: str) -> dict:
    import os, sys, json, subprocess, time
    from huggingface_hub import hf_hub_download, snapshot_download

    workspace = f"/tmp/uerf_{ckpt_name}"
    os.makedirs(workspace, exist_ok=True)

    # Download code files
    for fname in ["uerf_brain.py", "uerf_lifetime.py", "eval_checkpoint.py"]:
        hf_hub_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            filename=fname,
            local_dir=workspace,
            token=HF_TOKEN,
        )

    # Download checkpoint
    ckpt_path = os.path.join(workspace, ckpt_filename)
    print(f"[{ckpt_name}] Downloading checkpoint {ckpt_filename}...", flush=True)
    hf_hub_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        filename=ckpt_filename,
        local_dir=workspace,
        token=HF_TOKEN,
    )
    print(f"[{ckpt_name}] Download done ({os.path.getsize(ckpt_path)/1e9:.2f} GB)", flush=True)

    sys.path.insert(0, workspace)
    out_json = os.path.join(workspace, f"{ckpt_name}_eval.json")

    cmd = [
        sys.executable,
        os.path.join(workspace, "eval_checkpoint.py"),
        "--ckpt",    ckpt_path,
        "--n_eval",  "2000",
        "--n_calib", "300",
        "--n_relax", "8",
        "--seed",    "42",
        "--ceiling",
        "--out",     out_json,
    ]
    print(f"[{ckpt_name}] Running eval...", flush=True)
    result = subprocess.run(cmd, cwd=workspace)

    if result.returncode == 0 and os.path.exists(out_json):
        report = json.load(open(out_json))
        return report
    else:
        return {"error": f"eval failed with returncode={result.returncode}", "ckpt": ckpt_name}


@app.local_entrypoint()
def main():
    import json

    checkpoints = [
        ("emv2",     "emv2_main.pt"),
        ("pl1done",  "pl1done_main.pt"),
        ("emergent", "emergent_main.pt"),
    ]

    print("\nSubmitting eval jobs to Modal (A10G GPU)...\n")

    # Run all three in parallel
    results = {}
    for ckpt_name, result in zip(
        [c[0] for c in checkpoints],
        run_eval.starmap(checkpoints),
    ):
        results[ckpt_name] = result
        print(f"\n{'='*60}")
        print(f"  {ckpt_name}:")
        print(json.dumps(result, indent=4))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    fmt = "{:<12} {:>8} {:>12} {:>8} {:>12}"
    print(fmt.format("checkpoint", "raw%", "calibrated%", "top3%", "ceiling%"))
    print("-"*60)
    for name, r in results.items():
        if "error" not in r:
            print(fmt.format(
                name,
                f"{r.get('raw_acc', '?')}",
                f"{r.get('calibrated_acc', '?')}",
                f"{r.get('top3_acc', '?')}",
                f"{r.get('supervised_ceiling', '?')}",
            ))
        else:
            print(f"{name:<12}  ERROR: {r['error']}")
