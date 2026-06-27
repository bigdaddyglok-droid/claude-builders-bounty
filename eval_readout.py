#!/usr/bin/env python3
"""
eval_readout.py — Close the raw->probe gap. The default readout arg-maxes the
per-class score vector over ALL n_classes slots (310 after the full gauntlet),
even when the input is from one domain whose labels live in a known slot range.
Any louder out-of-domain slot steals the answer.

This script measures, for a given domain's class range [lo:hi], the accuracy of
each parameter-free readout channel BOTH unmasked (argmax over all slots) and
domain-masked (argmax restricted to [lo:hi]). If masking closes most of the gap
to the supervised probe, the knowledge was always present — the readout was
simply looking in the wrong drawer.

Channels (all parameter-free, no training):
  raw          norm argmax                ||s_teach||
  dir          signed projection argmax   (s_teach . c_teach)
  calib        per-slot z-scored norm     (||s|| - mu)/sd
  dir+calib    z-scored signed projection

GPU only. Matches the uerf_lifetime feature pipeline exactly.
"""
import argparse, json, time
import torch
import numpy as np
import uerf_brain as U

# domain -> (loader_name, class_offset, n_classes_in_domain)
DOMAINS = {
    'mnist': ('load_mnist',         0,   10),
    'cifar': ('load_cifar100',     10,  100),
    'ti':    ('load_tiny_imagenet',110, 200),
}


def build_features(loader_name, n_items, split, seed, device, class_offset):
    import uerf_lifetime as L
    encoder = L.UniversalEncoder(device=device)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=seed)
    proj.to(device)
    tr, te = getattr(L, loader_name)()
    ds = te if split == 'test' else tr
    xs, ys = [], []
    for i in range(min(n_items, len(ds))):
        img, lbl = ds[i]
        feat = encoder.encode(img.unsqueeze(0))[0]
        x = proj(feat)
        x = x / x.norm().clamp(min=1e-6)
        xs.append(x)
        ys.append(int(lbl) + class_offset)   # GLOBAL slot index
    return xs, ys


def capture(field, xs, n_relax):
    """Snapshot the 310-dim per-slot raw-norm and signed-projection vectors."""
    t_start, t_end = field.t_start, field.t_end
    c_teach = field.c[t_start:t_end]                      # (n_cls, d)
    raw_norms, dir_scores = [], []
    for x in xs:
        holder = {}
        orig = field.predict

        def probe(_orig=orig):
            s = field.s
            tv = s[t_start:t_end]
            holder['rawnorm'] = tv.norm(dim=-1).clone().cpu().numpy()
            holder['dir']     = (tv * c_teach).sum(-1).clone().cpu().numpy()
            return _orig()

        field.predict = probe
        field.eval_predict(x, n_relax=n_relax)
        field.predict = orig
        raw_norms.append(holder['rawnorm'])
        dir_scores.append(holder['dir'])
    return np.asarray(raw_norms), np.asarray(dir_scores)


def acc_masked(scores, y, lo, hi):
    """argmax restricted to slots [lo:hi], compared against global labels y."""
    sub = scores[:, lo:hi]
    pred = sub.argmax(1) + lo
    return 100.0 * (pred == y).mean()


def acc_unmasked(scores, y):
    return 100.0 * (scores.argmax(1) == y).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', default='ckpt')
    ap.add_argument('--domain', default='mnist', choices=list(DOMAINS))
    ap.add_argument('--n_eval',  type=int, default=2000)
    ap.add_argument('--n_calib', type=int, default=300)
    ap.add_argument('--n_relax', type=int, default=8)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out',     default='eval_readout.json')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    field = U.UERFField.load_checkpoint(args.ckpt, device=dev)
    loader, lo, ncls = DOMAINS[args.domain]
    hi = lo + ncls
    print(f"[{args.name}/{args.domain}] slots [{lo}:{hi}] of n_classes={field.n_classes} "
          f"exp={int(field.experience_count)} locked={int((field._class_locked & field.alive_mask).sum())}",
          flush=True)

    te_x, te_y = build_features(loader, args.n_eval, 'test', args.seed, dev, lo)
    te_y = np.asarray(te_y)
    n = len(te_y)
    print(f"[{args.name}] capturing {n} test samples...", flush=True)
    t0 = time.time()
    raw, dirs = capture(field, te_x, args.n_relax)
    print(f"[{args.name}] capture done in {time.time()-t0:.0f}s", flush=True)

    # calibration stats from unlabeled train inputs of the SAME domain
    cal_x, _ = build_features(loader, args.n_calib, 'train', args.seed, dev, lo)
    cal_raw, cal_dir = capture(field, cal_x, args.n_relax)
    mu_r, sd_r = cal_raw.mean(0), cal_raw.std(0).clip(min=1e-6)
    mu_d, sd_d = cal_dir.mean(0), cal_dir.std(0).clip(min=1e-6)

    raw_z = (raw - mu_r) / sd_r
    dir_z = (dirs - mu_d) / sd_d

    chance = 100.0 / ncls
    report = {
        'name': args.name, 'domain': args.domain, 'slots': [lo, hi],
        'n_eval': n, 'chance': round(chance, 2),
        'unmasked': {
            'raw':       round(acc_unmasked(raw, te_y), 1),
            'dir':       round(acc_unmasked(dirs, te_y), 1),
            'calib':     round(acc_unmasked(raw_z, te_y), 1),
            'dir_calib': round(acc_unmasked(dir_z, te_y), 1),
        },
        'domain_masked': {
            'raw':       round(acc_masked(raw, te_y, lo, hi), 1),
            'dir':       round(acc_masked(dirs, te_y, lo, hi), 1),
            'calib':     round(acc_masked(raw_z, te_y, lo, hi), 1),
            'dir_calib': round(acc_masked(dir_z, te_y, lo, hi), 1),
        },
    }
    json.dump(report, open(args.out, 'w'), indent=2)
    print(f"\n[{args.name}] READOUT REPORT:\n{json.dumps(report, indent=2)}", flush=True)
    return report


if __name__ == '__main__':
    main()
