"""Regenerate uerf_train_mnist.ipynb from the current uerf_brain.py + corrected harness."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAIN = open(os.path.join(ROOT, "uerf_brain.py")).read()

PIN = '''import subprocess, sys
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(f"GPU: {gpu}")
if "P100" in gpu:
    print("P100 detected — pinning torch 2.7.1 for Pascal (sm_60) support")
    subprocess.run(["pip", "install", "-q", "torch==2.7.1", "torchvision==0.22.1"], check=True)
    print("torch 2.7.1 installed")
import torch
print(f"torch {torch.__version__} | CUDA usable: {torch.cuda.is_available()}")
'''

HARNESS = r'''"""
UERF BRAIN — Kaggle Training Harness (corrected)
================================================
Split-MNIST continual-learning test on RAW pixels:
  Phase A : train digits 0-4   -> eval
  Phase B : train digits 5-9   -> re-eval A (retention) + B

Key corrections vs the earlier run:
- NO ablation toggles. The brain runs its proven behavior unconditionally.
- Zero-mean sensory encoding: the dataset mean image is subtracted before the
  random projection, removing the 80-97% common-mode that made dense digits
  argmax sinks. Discrimination is at the source — no readout correction.
- Domain-restricted retention: Phase-A retention is measured BOTH within 0-4
  (did it forget?) and across all 10 (real accuracy), so the number isn't
  conflated with inter-class competition.
"""
import os, sys, time, json, argparse
import torch
import numpy as np

from uerf_brain import UERFField, SensoryProjection

CONFIG = {
    'n_sensory'      : 64,
    'd'              : 32,
    'n_max'          : 400,
    'n_initial'      : 200,
    'n_relax'        : 6,
    'n_lock_per_cls' : 4,
    'phase_a_steps'  : 10000,
    'phase_b_steps'  : 10000,
    'eval_samples'   : 1000,
    'log_every'      : 200,
    'ckpt_every'     : 2000,
}

CKPT_DIR = '/kaggle/working/uerf_checkpoints' if os.path.isdir('/kaggle/working') else './uerf_checkpoints'
DATA_DIR = '/kaggle/working/data' if os.path.isdir('/kaggle/working') else './data'


def load_mnist():
    kaggle_train = '/kaggle/input/mnist-in-csv/mnist_train.csv'
    kaggle_test  = '/kaggle/input/mnist-in-csv/mnist_test.csv'
    if os.path.exists(kaggle_train) and os.path.exists(kaggle_test):
        print(f'[data] loading Kaggle MNIST CSV')
        tr = np.loadtxt(kaggle_train, delimiter=',', skiprows=1, dtype=np.float32)
        te = np.loadtxt(kaggle_test, delimiter=',', skiprows=1, dtype=np.float32)
        return (torch.from_numpy(tr[:, 1:]) / 255.0, torch.from_numpy(tr[:, 0]).long(),
                torch.from_numpy(te[:, 1:]) / 255.0, torch.from_numpy(te[:, 0]).long())
    print('[data] using torchvision MNIST')
    import torchvision
    from torchvision import transforms
    t = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True, transform=t)
    test  = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=t)
    train_imgs = torch.stack([s[0].view(-1) for s in train])
    train_lbls = torch.tensor([s[1] for s in train])
    test_imgs  = torch.stack([s[0].view(-1) for s in test])
    test_lbls  = torch.tensor([s[1] for s in test])
    return train_imgs, train_lbls, test_imgs, test_lbls


def evaluate(field, proj, imgs, labels, allowed_classes, domain, n_eval=1000):
    """Eval over inputs whose label is in allowed_classes, with the argmax
    restricted to `domain` = (lo, hi)."""
    allowed = set(int(c) for c in allowed_classes)
    lo, hi = domain
    correct = total = 0
    perm = torch.randperm(len(imgs))[:n_eval * 6]
    for idx in perm:
        if total >= n_eval:
            break
        y = int(labels[idx])
        if y not in allowed:
            continue
        x = proj(imgs[idx])
        pred = field.eval_predict(x, domain_lo=lo, domain_hi=hi)
        correct += int(pred == y)
        total += 1
    return (correct / total * 100.0) if total > 0 else 0.0


def train_phase(field, proj, imgs, labels, allowed_classes, n_steps, label,
                start_step=0, ckpt_path=None, log_every=200, ckpt_every=2000):
    print(f'\n[{label}] training {n_steps} steps on classes {allowed_classes}')
    allowed = set(int(c) for c in allowed_classes)
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for c in allowed:
        mask |= (labels == c)
    pool = mask.nonzero(as_tuple=True)[0]
    print(f'  phase pool size: {len(pool)}')
    rng = np.random.RandomState(start_step + 1)
    t0 = time.time()
    for step in range(start_step, start_step + n_steps):
        i = pool[rng.randint(0, len(pool))]
        y = int(labels[i])
        x = proj(imgs[i])
        tv = torch.zeros(field.n_classes, device=field.device)
        tv[y] = 1.0
        field.experience(x, teaching_vector=tv, n_relax=CONFIG['n_relax'], learn=True)
        if (step + 1) % log_every == 0:
            rate = (step - start_step + 1) / (time.time() - t0)
            r = field.report()
            top_state = max(r['state_dist'].items(), key=lambda kv: kv[1])[0] if r['state_dist'] else '?'
            print(f'  step {step+1:>6}  {rate:>5.1f} ex/s  bonds={r["bonds"]:>5}  '
                  f'alive={r["alive"]:>4}  top_state={top_state}', flush=True)
        if ckpt_path and (step + 1) % ckpt_every == 0:
            field.save_checkpoint(ckpt_path)
            print(f'  [ckpt] saved at step {step+1}', flush=True)
    if ckpt_path:
        field.save_checkpoint(ckpt_path)
    return start_step + n_steps


def run(quick=False):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_main = os.path.join(CKPT_DIR, 'main.pt')
    ckpt_phase_a = os.path.join(CKPT_DIR, 'after_phase_a.pt')
    log_path = os.path.join(CKPT_DIR, 'training_log.json')
    log = []

    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()
    print(f'[data] train {tuple(train_imgs.shape)}, test {tuple(test_imgs.shape)}')

    # Zero-mean sensory encoding: frozen dataset mean image (label-free), so the
    # common "average digit" DC component never enters bonds or readout.
    mean_img = train_imgs.mean(0)
    print(f'[encode] subtracting frozen dataset mean image (||mean||={mean_img.norm():.3f})')
    proj = SensoryProjection(784, CONFIG['n_sensory'], seed=42, mean_raw=mean_img)

    print('[init] new brain (no toggles — proven behavior is the only path)')
    torch.manual_seed(369)
    field = UERFField(n_max=CONFIG['n_max'], n_initial=CONFIG['n_initial'],
                      d=CONFIG['d'], input_dim=CONFIG['n_sensory'], n_classes=10)
    print(f'[brain] d={field.d}, alive={int(field.alive_mask.sum())}, bonds={int(field.C_mask.sum())}')

    if quick:
        CONFIG['phase_a_steps'] = 500
        CONFIG['phase_b_steps'] = 500
        CONFIG['eval_samples']  = 200
        print('[quick] small step counts')

    # ── Phase A: digits 0-4
    train_phase(field, proj, train_imgs, train_lbls, [0,1,2,3,4],
                CONFIG['phase_a_steps'], 'Phase A', start_step=0,
                ckpt_path=ckpt_main, log_every=CONFIG['log_every'],
                ckpt_every=CONFIG['ckpt_every'])
    field.save_checkpoint(ckpt_phase_a)

    print('\n[consolidate Phase A] locking digits 0-4 ...')
    n_locked = sum(field.consolidate_class(c, n_lock=CONFIG['n_lock_per_cls']) for c in [0,1,2,3,4])
    print(f'[consolidate Phase A] locked {n_locked} oscillators '
          f'(total locked={int(field._class_locked.sum().item())})')

    acc_a_init_r = evaluate(field, proj, test_imgs, test_lbls, [0,1,2,3,4], (0,5),  CONFIG['eval_samples'])
    acc_a_init_f = evaluate(field, proj, test_imgs, test_lbls, [0,1,2,3,4], (0,10), CONFIG['eval_samples'])
    print(f'\n[eval A] digits 0-4 — within-0-4: {acc_a_init_r:.1f}%   full-10way: {acc_a_init_f:.1f}%')
    log.append({'event': 'phase_a_complete', 'acc_a_within': acc_a_init_r, 'acc_a_full': acc_a_init_f})

    # ── Phase B: digits 5-9
    target = CONFIG['phase_a_steps'] + CONFIG['phase_b_steps']
    train_phase(field, proj, train_imgs, train_lbls, [5,6,7,8,9],
                target - field.experience_count, 'Phase B',
                start_step=field.experience_count, ckpt_path=ckpt_main,
                log_every=CONFIG['log_every'], ckpt_every=CONFIG['ckpt_every'])

    # ── Final eval
    acc_a_r = evaluate(field, proj, test_imgs, test_lbls, [0,1,2,3,4], (0,5),  CONFIG['eval_samples'])
    acc_a_f = evaluate(field, proj, test_imgs, test_lbls, [0,1,2,3,4], (0,10), CONFIG['eval_samples'])
    acc_b_r = evaluate(field, proj, test_imgs, test_lbls, [5,6,7,8,9], (5,10), CONFIG['eval_samples'])
    acc_b_f = evaluate(field, proj, test_imgs, test_lbls, [5,6,7,8,9], (0,10), CONFIG['eval_samples'])
    acc_all = evaluate(field, proj, test_imgs, test_lbls, list(range(10)), (0,10), CONFIG['eval_samples'])

    print('\n' + '='*64)
    print(' CATASTROPHIC FORGETTING REPORT  (zero-mean encoding, no toggles)')
    print('='*64)
    print(f'  Phase A within-0-4 : before B {acc_a_init_r:.1f}%  ->  after B {acc_a_r:.1f}%   (retention)')
    print(f'  Phase A full-10way : before B {acc_a_init_f:.1f}%  ->  after B {acc_a_f:.1f}%')
    print(f'  Phase B within-5-9 : {acc_b_r:.1f}%      full-10way: {acc_b_f:.1f}%')
    print(f'  All 10 digits      : {acc_all:.1f}%')
    print(f'  Chance: within-5way=20%, full-10way=10%')

    r = field.report()
    print(f'\n[brain] final state distribution: {r["state_dist"]}')
    print(f'[brain] bonds={r["bonds"]}, alive={r["alive"]}, births={r["births"]}, deaths={r["deaths"]}')

    log.append({'event': 'phase_b_complete',
                'acc_a_within_before': acc_a_init_r, 'acc_a_within_after': acc_a_r,
                'acc_a_full_before': acc_a_init_f, 'acc_a_full_after': acc_a_f,
                'acc_b_within': acc_b_r, 'acc_b_full': acc_b_f, 'acc_all': acc_all,
                'state_dist': r['state_dist'], 'bonds': r['bonds'],
                'experience_count': field.experience_count})
    with open(log_path, 'w') as f_:
        json.dump(log, f_, indent=2)
    print(f'\n[log] saved {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run(quick=args.quick)
'''

def cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

nb = {
    "cells": [
        cell(PIN),
        cell("%%writefile uerf_brain.py\n" + BRAIN),
        cell("%%writefile uerf_train.py\n" + HARNESS),
        cell("!python uerf_train.py 2>&1"),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i}"
out = os.path.join(os.path.dirname(__file__), "uerf_train_mnist.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "brain bytes:", len(BRAIN))
