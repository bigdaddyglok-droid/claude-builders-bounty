#!/usr/bin/env python3
"""
eval_full.py — FULL-CAPABILITY readout of a UERF checkpoint.

Does NOT settle for the magnitude-argmax readout. Queries every channel the
brain actually stores class identity in, and reports accuracy through each:

  CHANNEL 1  raw norm argmax          — what eval_predict() gives by default
  CHANNEL 2  calibrated norm          — training-free per-slot z-score de-bias
  CHANNEL 3  direction-aware teach    — argmax of signed projection s_teach·c_teach
                                        (uses DIRECTION, not just magnitude)
  CHANNEL 4  top-3 / top-5            — is the answer even in reach
  CHANNEL 5  PROBE: teach vectors     — linear readout of full d-dim teach state
  CHANNEL 6  PROBE: HOLOGRAPHIC plane — linear readout of the LOCKED oscillators'
                                        state (the "different dimensional plane")
  CHANNEL 7  PROBE: interior state    — linear readout of the whole interior field

Probes are train-on-half / test-on-other-half supervised linear readouts. They
measure how much class knowledge is PHYSICALLY PRESENT in that subspace — the
ceiling the architecture is capable of, independent of the lossy default readout.

GPU only. Matches the uerf_lifetime training feature pipeline exactly.
"""
import argparse, json, sys, time
import torch
import numpy as np

import uerf_brain as U


def build_eval_features(n_items, split, seed, device):
    import uerf_lifetime as L
    encoder = L.UniversalEncoder(device=device)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=seed)
    proj.to(device)
    tr, te = L.load_mnist()
    ds = te if split == 'test' else tr
    xs, ys = [], []
    for i in range(min(n_items, len(ds))):
        img, lbl = ds[i]
        feat = encoder.encode(img.unsqueeze(0))[0]
        x = proj(feat)
        x = x / x.norm().clamp(min=1e-6)
        xs.append(x)
        ys.append(int(lbl))
    return xs, ys


def capture_all(field, xs, n_relax):
    """For each input, run a non-destructive eval forward pass and snapshot
    the post-relaxation state from every readout channel."""
    t_start, t_end = field.t_start, field.t_end
    interior_mask = field.alive_mask & ~field._is_sensory & ~field._is_teaching
    interior_idx = interior_mask.nonzero(as_tuple=True)[0]
    locked_mask = field._class_locked & field.alive_mask
    locked_idx = locked_mask.nonzero(as_tuple=True)[0]
    c_teach = field.c[t_start:t_end]                      # (n_cls, d)

    teach_vecs, raw_norms, dir_scores, interior_vecs, locked_vecs = [], [], [], [], []

    for x in xs:
        holder = {}
        orig = field.predict

        def probe(_orig=orig):
            s = field.s
            tv = s[t_start:t_end]                          # (n_cls, d)
            holder['teach']    = tv.flatten().clone().cpu().numpy()
            holder['rawnorm']  = tv.norm(dim=-1).clone().cpu().numpy()
            holder['dir']      = (tv * c_teach).sum(-1).clone().cpu().numpy()
            holder['interior'] = s[interior_idx].flatten().clone().cpu().numpy()
            if len(locked_idx) > 0:
                holder['locked'] = s[locked_idx].flatten().clone().cpu().numpy()
            return _orig()

        field.predict = probe
        field.eval_predict(x, n_relax=n_relax)
        field.predict = orig

        teach_vecs.append(holder['teach'])
        raw_norms.append(holder['rawnorm'])
        dir_scores.append(holder['dir'])
        interior_vecs.append(holder['interior'])
        if 'locked' in holder:
            locked_vecs.append(holder['locked'])

    return {
        'teach':    np.asarray(teach_vecs),
        'rawnorm':  np.asarray(raw_norms),
        'dir':      np.asarray(dir_scores),
        'interior': np.asarray(interior_vecs),
        'locked':   np.asarray(locked_vecs) if locked_vecs else None,
        'n_locked': len(locked_idx),
        'n_interior': len(interior_idx),
    }


def probe_acc(X, y, C=1.0):
    """Train-on-half / test-on-half supervised linear readout accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    split = n // 2
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(Xtr), ytr)
    return 100.0 * clf.score(sc.transform(Xte), yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', default='ckpt')
    ap.add_argument('--n_eval',  type=int, default=2000)
    ap.add_argument('--n_calib', type=int, default=300)
    ap.add_argument('--n_relax', type=int, default=8)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--out',     default='eval_full.json')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{args.name}] device={dev}  loading {args.ckpt} ...", flush=True)
    field = U.UERFField.load_checkpoint(args.ckpt, device=dev)

    n_locked_total = int((field._class_locked & field.alive_mask).sum())
    n_interior_total = int((field.alive_mask & ~field._is_sensory & ~field._is_teaching).sum())
    print(f"[{args.name}] n_max={field.n_max} d={field.d} input_dim={field.input_dim} "
          f"n_classes={field.n_classes} exp={int(field.experience_count)} "
          f"locked={n_locked_total} interior={n_interior_total}", flush=True)

    # ── Capture every channel on the test set ──────────────────────────────────
    te_x, te_y = build_eval_features(args.n_eval, 'test', args.seed, dev)
    te_y = np.asarray(te_y)
    n = len(te_y)
    print(f"[{args.name}] capturing {n} test samples across all channels...", flush=True)
    t0 = time.time()
    cap = capture_all(field, te_x, args.n_relax)
    print(f"[{args.name}] capture done in {time.time()-t0:.0f}s", flush=True)

    # ── Calibration stats from unlabeled train inputs ───────────────────────────
    cal_x, _ = build_eval_features(args.n_calib, 'train', args.seed, dev)
    cal_cap = capture_all(field, cal_x, args.n_relax)
    mu = cal_cap['rawnorm'].mean(0)
    sd = cal_cap['rawnorm'].std(0).clip(min=1e-6)

    # ── CHANNEL accuracies ──────────────────────────────────────────────────────
    raw_pred = cap['rawnorm'].argmax(1)
    raw_acc  = 100.0 * (raw_pred == te_y).mean()

    cal_pred = ((cap['rawnorm'] - mu) / sd).argmax(1)
    cal_acc  = 100.0 * (cal_pred == te_y).mean()

    dir_pred = cap['dir'].argmax(1)
    dir_acc  = 100.0 * (dir_pred == te_y).mean()

    # top-k from raw norms
    order = np.argsort(-cap['rawnorm'], axis=1)
    top3 = 100.0 * np.mean([te_y[i] in order[i, :3] for i in range(n)])
    top5 = 100.0 * np.mean([te_y[i] in order[i, :5] for i in range(n)])

    report = {
        'name': args.name, 'ckpt': args.ckpt, 'n_eval': n,
        'dims': {'n_max': field.n_max, 'd': field.d,
                 'input_dim': field.input_dim, 'n_classes': field.n_classes,
                 'locked': n_locked_total, 'interior': n_interior_total,
                 'experience': int(field.experience_count)},
        'ch1_raw_norm_argmax':   round(raw_acc, 1),
        'ch2_calibrated_norm':   round(cal_acc, 1),
        'ch3_direction_teach':   round(dir_acc, 1),
        'ch4_top3':              round(top3, 1),
        'ch4_top5':              round(top5, 1),
    }

    # ── PROBES (capability ceilings per subspace) ───────────────────────────────
    print(f"[{args.name}] probing teach-vector subspace...", flush=True)
    report['ch5_probe_teach_vectors'] = round(probe_acc(cap['teach'], te_y), 1)

    if cap['locked'] is not None:
        print(f"[{args.name}] probing HOLOGRAPHIC locked plane ({cap['n_locked']} osc)...", flush=True)
        report['ch6_probe_holographic_plane'] = round(probe_acc(cap['locked'], te_y), 1)
    else:
        report['ch6_probe_holographic_plane'] = None  # no locked osc (un-consolidated)

    print(f"[{args.name}] probing interior state ({cap['n_interior']} osc)...", flush=True)
    # interior is high-dim; regularize harder to avoid overfit
    report['ch7_probe_interior_state'] = round(probe_acc(cap['interior'], te_y, C=0.05), 1)

    json.dump(report, open(args.out, 'w'), indent=2)
    print(f"\n[{args.name}] FULL REPORT:\n{json.dumps(report, indent=2)}", flush=True)
    return report


if __name__ == '__main__':
    main()
