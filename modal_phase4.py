"""
modal_phase4.py — Final Lifetime Gauntlet evaluation (Phase 4, read-only).

Loads the Phase 3 checkpoint (phase3_main.pt from HF) and runs eval_phase
across all three domains: MNIST (10 cls), CIFAR-100 (100 cls), TinyImageNet
(200 cls). No training — pure memory / retention test.

Uses the exact same brain + lifetime code Phase 3 ran with (phase2_brain.py /
phase2_lifetime.py) for checkpoint compatibility.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-phase4", image=image)

HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run_phase4(hf_token: str):
    import os, sys, time, json, shutil
    from huggingface_hub import hf_hub_download, HfApi

    api = HfApi(token=hf_token)
    BASE = "/uerf-output"
    CKPT_DIR = os.path.join(BASE, "lifetime_checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    work = "/root/work"
    os.makedirs(work, exist_ok=True)

    def grab(fn, dest_dir, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename=fn, local_dir=dest_dir, token=hf_token)
        if rename:
            target = os.path.join(dest_dir, rename)
            shutil.copy(p, target)
            return target
        return p

    print("[setup] downloading Phase 3 code + checkpoint from HF...", flush=True)
    # Use exact same brain/lifetime code Phase 3 ran with for checkpoint compatibility
    grab("phase2_brain.py",    work, rename="uerf_brain.py")
    grab("phase2_lifetime.py", work, rename="uerf_lifetime.py")
    ckpt_src = grab("phase3_main.pt", work)
    shutil.copy(ckpt_src, os.path.join(CKPT_DIR, "lifetime_main.pt"))
    log_src = grab("phase3_log.json", work)
    shutil.copy(log_src, os.path.join(CKPT_DIR, "lifetime_log.json"))
    print(f"[setup] checkpoint in place: {os.path.getsize(os.path.join(CKPT_DIR,'lifetime_main.pt'))/1e9:.2f} GB", flush=True)

    env = dict(os.environ)
    env["UERF_NMAX"] = "800"
    env["UERF_D"] = "32"
    env["PYTHONUNBUFFERED"] = "1"

    import subprocess
    print("[phase4] launching: python uerf_lifetime.py --phase 4", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "uerf_lifetime.py", "--phase", "4"],
        cwd=work, env=env,
    )
    dt = time.time() - t0
    print(f"[phase4] process exited rc={proc.returncode} in {dt/60:.1f} min", flush=True)

    final_log = os.path.join(CKPT_DIR, "lifetime_log.json")
    result = {"returncode": proc.returncode, "eval_minutes": round(dt / 60, 1)}
    if os.path.exists(final_log):
        try:
            log = json.load(open(final_log))
            result["log"] = log
            # surface the phase_4_final event
            for entry in reversed(log):
                if entry.get("event") == "phase_4_final":
                    result["final"] = entry
                    break
        except Exception:
            pass
        api.upload_file(path_or_fileobj=final_log, path_in_repo="phase4_log.json",
                        repo_id=HF_DATASET, repo_type="dataset")
        print("[phase4] uploaded phase4_log.json", flush=True)
    return result
