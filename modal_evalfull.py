"""
modal_evalfull.py — Modal app (importable module) for the full multi-channel
UERF readout. Defined as a module with NO top-level app.run() so remote
containers can import it cleanly. Drive it from a separate runner that does
`with app.run(): main(...)`.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-eval-full2", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=14000, cpu=8.0)
def run_full(name: str, ckpt_filename: str, hf_token: str) -> dict:
    import os, sys, json, subprocess
    from huggingface_hub import hf_hub_download
    ws = f"/tmp/uerf_{name}"
    os.makedirs(ws, exist_ok=True)
    for f in ["uerf_brain.py", "uerf_lifetime.py", "eval_full.py", ckpt_filename]:
        hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=f,
                        local_dir=ws, token=hf_token)
    ckpt = os.path.join(ws, ckpt_filename)
    out = os.path.join(ws, f"{name}_full.json")
    cmd = [sys.executable, os.path.join(ws, "eval_full.py"),
           "--ckpt", ckpt, "--name", name, "--n_eval", "2000",
           "--n_calib", "300", "--n_relax", "8", "--seed", "42", "--out", out]
    r = subprocess.run(cmd, cwd=ws)
    if r.returncode == 0 and os.path.exists(out):
        return json.load(open(out))
    return {"error": f"returncode={r.returncode}", "name": name}


def main(token):
    import json
    CKPTS = [("emv2", "emv2_main.pt"),
             ("pl1done", "pl1done_main.pt"),
             ("emergent", "emergent_main.pt")]
    args = [(n, f, token) for n, f in CKPTS]
    results = {}
    for (name, _, _), res in zip(args, run_full.starmap(args)):
        results[name] = res
        print(f"\n{'='*64}\n{name}:\n{json.dumps(res, indent=2)}", flush=True)
    print("\n\n" + "=" * 64)
    print("FULL CAPABILITY SUMMARY — accuracy through every readout channel")
    print("=" * 64)
    hdr = ["checkpoint", "ch1raw", "ch2calib", "ch3dir", "top3", "top5",
           "ch5teach", "ch6holo", "ch7inter"]
    print(" | ".join(f"{h:>10}" for h in hdr))
    print("-" * 110)
    for name, r in results.items():
        if "error" in r:
            print(f"{name}: ERROR {r['error']}"); continue
        row = [name, r.get('ch1_raw_norm_argmax'), r.get('ch2_calibrated_norm'),
               r.get('ch3_direction_teach'), r.get('ch4_top3'), r.get('ch4_top5'),
               r.get('ch5_probe_teach_vectors'), r.get('ch6_probe_holographic_plane'),
               r.get('ch7_probe_interior_state')]
        print(" | ".join(f"{str(c):>10}" for c in row))
    return results
