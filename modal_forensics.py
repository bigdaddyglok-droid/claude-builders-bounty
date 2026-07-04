"""
modal_forensics.py — run the existing uerf_forensics.py dissection on the real
trained checkpoints, on Modal (the .pt lives on HF; nothing staged locally).
Returns the full forensic report text + JSON + the visual dashboard PNG(s).
Drive: with app.run(): main(["phase3_main.pt", ...])
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy", "matplotlib", "scikit-learn",
                 "huggingface_hub", "pillow")
)
app = modal.App("uerf-forensics", image=image)
HF = "BlackLoks/uerf-checkpoints"


@app.function(timeout=3600, memory=40000, cpu=8.0)
def run_forensics(ckpt_filename: str, forensics_src: str, hf_token: str) -> dict:
    import os, sys, subprocess, json, glob, base64, shutil
    os.environ["MPLBACKEND"] = "Agg"
    from huggingface_hub import hf_hub_download
    ws = "/tmp/ws"; os.makedirs(ws, exist_ok=True)

    def grab(fn, rn=None):
        p = hf_hub_download(HF, fn, repo_type="dataset", token=hf_token, local_dir=ws)
        if rn: shutil.copy(p, os.path.join(ws, rn))
        return os.path.join(ws, rn or fn)

    grab("phase2_brain.py", "uerf_brain.py")
    grab(ckpt_filename)
    open(os.path.join(ws, "uerf_forensics.py"), "w").write(forensics_src)
    out = os.path.join(ws, "fout"); os.makedirs(out, exist_ok=True)

    cmd = [sys.executable, os.path.join(ws, "uerf_forensics.py"),
           os.path.join(ws, ckpt_filename), "--visualize",
           "--json", os.path.join(out, "report.json"), "--out-dir", out]
    r = subprocess.run(cmd, cwd=ws, capture_output=True, text=True)
    res = {"ckpt": ckpt_filename, "rc": r.returncode,
           "stdout": r.stdout, "stderr": r.stderr[-4000:]}
    jp = os.path.join(out, "report.json")
    if os.path.exists(jp):
        try: res["report"] = json.load(open(jp))
        except Exception as e: res["report_err"] = str(e)
    pngs = {}
    for p in sorted(glob.glob(os.path.join(out, "*.png"))):
        pngs[os.path.basename(p)] = base64.b64encode(open(p, "rb").read()).decode()
    res["pngs"] = pngs
    return res


def main(ckpts=None):
    import os, base64
    tok = os.environ["HF_READ_TOKEN"]
    src = open(os.path.join(os.path.dirname(__file__), "uerf_forensics.py")).read()
    ckpts = ckpts or ["phase3_main.pt"]
    saved = []
    for ck in ckpts:
        print(f"\n{'#'*70}\n# FORENSICS: {ck}\n{'#'*70}", flush=True)
        res = run_forensics.remote(ck, src, tok)
        print(res["stdout"], flush=True)
        if res.get("stderr"): print("STDERR:", res["stderr"][-1500:], flush=True)
        for name, b64 in res.get("pngs", {}).items():
            fn = f"forensics_{ck.replace('.pt','')}_{name}"
            open(fn, "wb").write(base64.b64decode(b64))
            saved.append(fn); print(f"[saved dashboard] {fn}", flush=True)
    print("\nSAVED FILES:", saved, flush=True)
    return saved
