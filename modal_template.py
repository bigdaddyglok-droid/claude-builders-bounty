"""
modal_template.py — Run eval_template.py (nearest-prototype template readout)
on the Phase 3 checkpoint across all three domains on Modal GPU.
Clean module, NO top-level app.run().
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "numpy",
                 "scikit-learn", "huggingface_hub", "pillow")
)
app = modal.App("uerf-template", image=image)
HF_DATASET = "BlackLoks/uerf-checkpoints"


@app.function(gpu="A10G", timeout=7200, memory=46080, cpu=8.0)
def run_template(domain: str, ckpt_filename: str, hf_token: str) -> dict:
    import os, sys, json, subprocess, shutil
    from huggingface_hub import hf_hub_download
    ws = f"/tmp/uerf_tmpl_{domain}"
    os.makedirs(ws, exist_ok=True)

    def grab(fn, rename=None):
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset", filename=fn,
                            local_dir=ws, token=hf_token)
        if rename:
            t = os.path.join(ws, rename); shutil.copy(p, t); return t
        return p

    grab("uerf_brain.py")          # new brain with fit_teach_templates
    grab("phase2_lifetime.py", rename="uerf_lifetime.py")
    grab("eval_template.py")
    grab(ckpt_filename)

    ckpt = os.path.join(ws, ckpt_filename)
    out  = os.path.join(ws, f"tmpl_{domain}.json")
    cmd  = [sys.executable, os.path.join(ws, "eval_template.py"),
            "--ckpt", ckpt, "--name", "phase3", "--domain", domain,
            "--n_eval", "2000", "--n_calib", "500",
            "--n_relax", "8", "--seed", "42", "--out", out]
    r = subprocess.run(cmd, cwd=ws)
    if r.returncode == 0 and os.path.exists(out):
        return json.load(open(out))
    return {"error": f"returncode={r.returncode}", "domain": domain}


def main(token):
    import json
    domains = ["mnist", "cifar", "ti"]
    args = [(d, "phase3_main.pt", token) for d in domains]
    results = {}
    for (d, _, _), res in zip(args, run_template.starmap(args)):
        results[d] = res
        print(f"\n{'='*64}\n{d}:\n{json.dumps(res, indent=2)}", flush=True)

    print("\n\n" + "=" * 64)
    print("NEAREST-PROTOTYPE TEMPLATE READOUT — Phase 3 checkpoint")
    print("=" * 64)
    hdr = ["domain", "chance", "base(cal)", "slot(unmsk)", "slot(mask)", "field(unmsk)", "field(mask)", "probe_ceil"]
    print(" | ".join(f"{h:>12}" for h in hdr))
    print("-" * 110)
    for d, r in results.items():
        if "error" in r:
            print(f"{d}: ERROR {r['error']}"); continue
        row = [d, r['chance'], r['baseline_masked_calib'],
               r['template_unmasked'], r['template_masked'],
               r.get('field_tmpl_unmasked', 'n/a'), r.get('field_tmpl_masked', 'n/a'),
               r.get('probe_ceiling_mnist', 'n/a')]
        print(" | ".join(f"{str(c):>12}" for c in row))
    return results
