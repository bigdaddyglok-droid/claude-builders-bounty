# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.4.0",
#   "torchvision==0.19.0",
#   "numpy",
#   "scikit-learn",
#   "huggingface_hub",
# ]
# ///
"""
HF Job: Download UERF checkpoints from BlackLoks/uerf-checkpoints dataset,
run eval_checkpoint.py on each, and print results.

Runs on: a10g-small (24 GB VRAM, 15 GB RAM, $0.016/min)
"""
import os, sys, json, subprocess, time
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get('HF_TOKEN', '')
WORKSPACE = '/tmp/uerf_workspace'
os.makedirs(WORKSPACE, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Downloading dataset BlackLoks/uerf-checkpoints ...", flush=True)
data_dir = snapshot_download(
    repo_id='BlackLoks/uerf-checkpoints',
    repo_type='dataset',
    local_dir=WORKSPACE,
    token=HF_TOKEN,
)
print(f"[{time.strftime('%H:%M:%S')}] Dataset downloaded to {data_dir}", flush=True)

# Add workspace to Python path so uerf_brain / uerf_lifetime can be imported
sys.path.insert(0, WORKSPACE)

# The three consolidated checkpoints (most trained first)
CKPTS = [
    ('emv2',    os.path.join(WORKSPACE, 'emv2_main.pt')),
    ('pl1done', os.path.join(WORKSPACE, 'pl1done_main.pt')),
    ('emergent',os.path.join(WORKSPACE, 'emergent_main.pt')),
]

eval_script = os.path.join(WORKSPACE, 'eval_checkpoint.py')
all_results = {}

for name, ckpt_path in CKPTS:
    if not os.path.exists(ckpt_path):
        print(f"[{time.strftime('%H:%M:%S')}] SKIP {name}: not found at {ckpt_path}", flush=True)
        continue

    out_json = os.path.join(WORKSPACE, f'{name}_eval.json')
    print(f"\n{'='*60}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Evaluating: {name}  ({os.path.getsize(ckpt_path)/1e9:.2f} GB)", flush=True)

    cmd = [
        sys.executable, eval_script,
        '--ckpt',    ckpt_path,
        '--n_eval',  '2000',
        '--n_calib', '300',
        '--n_relax', '8',
        '--seed',    '42',
        '--ceiling',
        '--out',     out_json,
    ]
    result = subprocess.run(cmd, cwd=WORKSPACE)
    if result.returncode == 0 and os.path.exists(out_json):
        report = json.load(open(out_json))
        all_results[name] = report
        print(f"[{time.strftime('%H:%M:%S')}] {name} DONE:", flush=True)
        print(json.dumps(report, indent=2), flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {name} FAILED (returncode={result.returncode})", flush=True)

print(f"\n{'='*60}", flush=True)
print("FINAL RESULTS SUMMARY", flush=True)
print('='*60, flush=True)
for name, r in all_results.items():
    print(f"\n{name}:", flush=True)
    print(f"  raw argmax acc      : {r.get('raw_acc')}%", flush=True)
    print(f"  calibrated acc      : {r.get('calibrated_acc')}%  "
          f"({r.get('calibrated_acc',0)-r.get('raw_acc',0):+.1f})", flush=True)
    print(f"  top-3 acc           : {r.get('top3_acc')}%", flush=True)
    if 'supervised_ceiling' in r:
        print(f"  supervised ceiling  : {r['supervised_ceiling']}%  "
              f"(knowledge gap = {r['supervised_ceiling']-r.get('calibrated_acc',0):.1f}%)", flush=True)

# Write combined results
combined_path = os.path.join(WORKSPACE, 'all_results.json')
json.dump(all_results, open(combined_path, 'w'), indent=2)
print(f"\nCombined results: {combined_path}", flush=True)
