"""
modal_phase3.py — Run UERF Lifetime Gauntlet Phase 3 (TinyImageNet, 200 classes)
on Modal GPU, continuing from the completed Phase 2 (CIFAR-100) checkpoint.

Uses the EXACT brain + lifetime code that Phase 2 ran with (phase2_brain.py /
phase2_lifetime.py from the HF dataset) for lifetime continuity. Config matches
the checkpoint: UERF_NMAX=800, UERF_D=32.

A background thread backs up the in-progress checkpoint to HF every 15 min so a
multi-hour run can resume after any interruption.
"""
import ssl, os

# SSL patch for Anthropic egress proxy (this container only)
_orig_ssl = ssl.create_default_context
def _patched_ssl(*a, **k):
    ctx = _orig_ssl(*a, **k)
    try: ctx.load_verify_locations('/tmp/anthropic_root_ca.pem')
    except Exception: pass
    return ctx
ssl.create_default_context = _patched_ssl

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-phase3", image=image)

HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=21600, memory=46080, cpu=12.0)
def train_phase3(hf_token: str):
    import os, sys, time, json, shutil, threading
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi(token=hf_token)
    BASE = "/uerf-output"
    CKPT_DIR = os.path.join(BASE, "lifetime_checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    work = "/root/work"
    os.makedirs(work, exist_ok=True)

    # ── Pull the EXACT Phase 2 code + checkpoint ────────────────────────────────
    def grab(fn, dest_dir, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename=fn, local_dir=dest_dir, token=hf_token)
        if rename:
            target = os.path.join(dest_dir, rename)
            shutil.copy(p, target)
            return target
        return p

    print("[setup] downloading Phase 2 code + checkpoint from HF...", flush=True)
    grab("phase2_brain.py",    work, rename="uerf_brain.py")
    grab("phase2_lifetime.py", work, rename="uerf_lifetime.py")
    ckpt_src = grab("phase2_main.pt", work)
    shutil.copy(ckpt_src, os.path.join(CKPT_DIR, "lifetime_main.pt"))
    log_src = grab("phase2_log.json", work)
    shutil.copy(log_src, os.path.join(CKPT_DIR, "lifetime_log.json"))
    print(f"[setup] checkpoint in place: {os.path.getsize(os.path.join(CKPT_DIR,'lifetime_main.pt'))/1e9:.2f} GB", flush=True)

    # ── Config to match the checkpoint (n_max=800, d=32) ────────────────────────
    env = dict(os.environ)
    env["UERF_NMAX"] = "800"
    env["UERF_D"] = "32"
    env["PYTHONUNBUFFERED"] = "1"

    # ── Periodic HF backup of the in-progress checkpoint ────────────────────────
    stop_flag = {"stop": False}
    def backup_loop():
        ckpt_path = os.path.join(CKPT_DIR, "lifetime_main.pt")
        last_mtime = 0
        while not stop_flag["stop"]:
            for _ in range(90):  # ~15 min, check stop flag every 10s
                if stop_flag["stop"]:
                    break
                time.sleep(10)
            try:
                if os.path.exists(ckpt_path):
                    m = os.path.getmtime(ckpt_path)
                    if m != last_mtime:
                        last_mtime = m
                        api.upload_file(path_or_fileobj=ckpt_path,
                                        path_in_repo="phase3_progress.pt",
                                        repo_id=HF_DATASET, repo_type="dataset")
                        print(f"[backup] pushed phase3_progress.pt @ {time.strftime('%H:%M:%S')}", flush=True)
            except Exception as e:
                print(f"[backup] failed (non-fatal): {e}", flush=True)
    bt = threading.Thread(target=backup_loop, daemon=True)
    bt.start()

    # ── Run Phase 3 training ────────────────────────────────────────────────────
    import subprocess
    print("[phase3] launching: python uerf_lifetime.py --phase 3", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "uerf_lifetime.py", "--phase", "3"],
        cwd=work, env=env,
    )
    dt = time.time() - t0
    print(f"[phase3] training process exited rc={proc.returncode} in {dt/60:.1f} min", flush=True)

    stop_flag["stop"] = True
    time.sleep(2)

    # ── Final upload of the resulting checkpoint + log ──────────────────────────
    final_ckpt = os.path.join(CKPT_DIR, "lifetime_main.pt")
    final_log = os.path.join(CKPT_DIR, "lifetime_log.json")
    result = {"returncode": proc.returncode, "train_minutes": round(dt / 60, 1)}
    if os.path.exists(final_ckpt):
        api.upload_file(path_or_fileobj=final_ckpt, path_in_repo="phase3_main.pt",
                        repo_id=HF_DATASET, repo_type="dataset")
        print("[phase3] uploaded phase3_main.pt", flush=True)
        result["uploaded"] = "phase3_main.pt"
    if os.path.exists(final_log):
        api.upload_file(path_or_fileobj=final_log, path_in_repo="phase3_log.json",
                        repo_id=HF_DATASET, repo_type="dataset")
        # surface the final scores
        try:
            result["log"] = json.load(open(final_log))
        except Exception:
            pass
    return result


@app.local_entrypoint()
def main():
    import json
    token = os.environ.get("HF_TOKEN", "")  # set via env, never hardcode
    if not token:
        raise SystemExit("Set HF_TOKEN env var before running.")
    print("Launching Phase 3 (TinyImageNet, 200 classes, 40K steps) on Modal A10G...")
    res = train_phase3.remote(token)
    print("\n" + "=" * 60)
    print("PHASE 3 RESULT")
    print("=" * 60)
    print(json.dumps(res, indent=2))
