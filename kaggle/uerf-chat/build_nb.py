"""Builds uerf_chat.ipynb — char-level chatbot/LM on the oscillator field."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAIN = open(os.path.join(ROOT, "uerf_brain.py")).read()
HARNESS = open(os.path.join(os.path.dirname(__file__), "harness.py")).read()

PIN = '''import subprocess, sys
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu}")
if "P100" in gpu:
    subprocess.run(["pip","install","-q","torch==2.7.1","torchvision==0.22.1"], check=True)
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
'''

def cell(src, cid):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"id":cid,
            "source":src.splitlines(keepends=True)}

nb={"cells":[cell(PIN,"pin"),
             cell("%%writefile uerf_brain.py\n"+BRAIN,"brain"),
             cell("%%writefile harness.py\n"+HARNESS,"harness"),
             cell("!UERF_N_TRAIN=12000 UERF_REPORT=1500 UERF_SLICE=25000 python harness.py 2>&1","run")],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"uerf_chat.ipynb")
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"brain",len(BRAIN),"harness",len(HARNESS))
