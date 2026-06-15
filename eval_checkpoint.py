#!/usr/bin/env python3
"""
eval_checkpoint.py — interpret ANY UERF main.pt and report the REAL test results.

What it does
------------
1. Loads ANY checkpoint via UERFField.load_checkpoint (auto-detects n_max/d/
   input_dim/n_classes from the .pt — NO CONFIG dependency, so the
   "Checkpoint dims don't match CONFIG" error cannot happen here).
2. Applies TRAINING-FREE z-score readout calibration (recovers the accuracy the
   raw norm-argmax readout throws away — verified +~10 pts at proxy scale).
3. Reports, for the checkpoint:
      - raw argmax accuracy           (what eval_predict currently gives)
      - calibrated accuracy           (training-free, the real readout number)
      - top-3 accuracy                (how often the answer is even in reach)
      - [--ceiling] supervised PROBE   (how much class knowledge is ACTUALLY in
        the brain's state — the gap between this and 'calibrated' is the
        knowledge we still aren't communicating)

CRITICAL — the feature pipeline must match training
---------------------------------------------------
The brain was trained on features from UniversalEncoder (frozen ResNet-18,
512-d) → SensoryProjection(512, n_sensory=128, seed=42). This script reuses
those exact components from uerf_lifetime so features match training exactly.

Usage
-----
    python eval_checkpoint.py --ckpt lifetime_main.pt
    python eval_checkpoint.py --ckpt lifetime_main.pt --n_eval 2000 --n_calib 300
    python eval_checkpoint.py --ckpt lifetime_main.pt --ceiling

    # Run all Phase-1 checkpoints:
    for ckpt in /tmp/emv2/lifetime_checkpoints/lifetime_main.pt \
                /tmp/pl1done/lifetime_checkpoints/lifetime_main.pt \
                /tmp/emergent/lifetime_checkpoints/lifetime_main.pt; do
        python eval_checkpoint.py --ckpt $ckpt --ceiling --out $(basename $(dirname $(dirname $ckpt)))_eval.json
    done
"""
import argparse, json, sys, time
import torch

import uerf_brain as U   # your brain module (must be importable / same dir)


# ============================================================================
# FEATURE PIPELINE  —  must exactly match uerf_lifetime training pipeline.
# Uses UniversalEncoder (frozen ResNet-18 → 512-d) + SensoryProjection(512→128).
# Returns (list_of_feature_tensors, list_of_labels).
# ============================================================================
def build_eval_features(n_items, split='test', seed=42, device='cpu'):
    import uerf_lifetime as L

    # Encoder: frozen ResNet-18, same as training
    encoder = L.UniversalEncoder(device=device)

    # Projection: same dim (512→128) and seed as train_phase / eval_phase
    n_sensory = L.CONFIG['n_sensory']   # 128
    proj = U.SensoryProjection(512, n_sensory, seed=seed)
    proj.to(device)

    # Data: MNIST train or test split
    tr, te = L.load_mnist()
    ds = te if split == 'test' else tr

    xs, ys = [], []
    for i in range(min(n_items, len(ds))):
        img, lbl = ds[i]
        feat = encoder.encode(img.unsqueeze(0))[0]   # (512,) unit-normed
        x = proj(feat)                                # (n_sensory,)
        # eval_predict normalises internally, but be consistent with training
        x = x / x.norm().clamp(min=1e-6)
        xs.append(x)
        ys.append(int(lbl))
    return xs, ys
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n_eval',  type=int, default=2000)
    ap.add_argument('--n_calib', type=int, default=300)
    ap.add_argument('--n_relax', type=int, default=8)
    ap.add_argument('--seed',    type=int, default=42,
                    help='SensoryProjection seed — MUST match training')
    ap.add_argument('--ceiling', action='store_true',
                    help='also fit a supervised probe to show knowledge ceiling')
    ap.add_argument('--out',     default='eval_report.json')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  loading {args.ckpt} ...", flush=True)
    field = U.UERFField.load_checkpoint(args.ckpt, device=dev)
    field._replay_disabled = True
    print(f"loaded: n_max={field.n_max} d={field.d} input_dim={field.input_dim} "
          f"n_classes={field.n_classes} exp_count={int(field.experience_count)}", flush=True)

    # data
    te_x, te_y = build_eval_features(args.n_eval, 'test', args.seed, device=dev)
    print(f"test items: {len(te_x)}", flush=True)

    # 1) RAW accuracy (uncalibrated)
    field._readout_mu = None
    field._readout_sigma = None
    raw_correct = 0; top3 = 0
    te_scores = []
    t0 = time.time()
    for i, (x, y) in enumerate(zip(te_x, te_y)):
        r, sc = field.eval_predict(x, n_relax=args.n_relax, return_scores=True)
        te_scores.append(sc)
        if r == y:
            raw_correct += 1
        if y in torch.topk(sc, min(3, sc.numel())).indices.tolist():
            top3 += 1
        if (i + 1) % 100 == 0:
            print(f"  raw pass {i+1}/{len(te_x)}  running_acc="
                  f"{100*raw_correct/(i+1):.1f}%  elapsed={time.time()-t0:.0f}s", flush=True)
    n = len(te_y)
    raw_acc   = 100 * raw_correct / n
    top3_acc  = 100 * top3 / n
    print(f"raw argmax acc = {raw_acc:.1f}%   top3 = {top3_acc:.1f}%", flush=True)

    # 2) TRAINING-FREE calibration on UNLABELED inputs, then re-eval
    calib_x, _ = build_eval_features(args.n_calib, 'train', args.seed, device=dev)
    field.calibrate_readout(calib_x, n_relax=args.n_relax)
    cal_correct = 0
    for i, (x, y) in enumerate(zip(te_x, te_y)):
        if field.eval_predict(x, n_relax=args.n_relax) == y:
            cal_correct += 1
        if (i + 1) % 100 == 0:
            print(f"  cal pass {i+1}/{len(te_x)}  running_acc="
                  f"{100*cal_correct/(i+1):.1f}%", flush=True)
    cal_acc = 100 * cal_correct / n
    print(f"calibrated acc = {cal_acc:.1f}%  (training-free, {cal_acc-raw_acc:+.1f})", flush=True)

    report = {
        'ckpt':           args.ckpt,
        'n_eval':         n,
        'raw_acc':        round(raw_acc, 1),
        'calibrated_acc': round(cal_acc, 1),
        'top3_acc':       round(top3_acc, 1),
        'dims': {
            'n_max':      field.n_max,
            'd':          field.d,
            'input_dim':  field.input_dim,
            'n_classes':  field.n_classes,
        },
    }

    # 3) optional SUPERVISED ceiling — how much knowledge is actually in there
    if args.ceiling:
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            field._readout_mu = None
            field._readout_sigma = None

            # Capture pre-readout teach-slot vectors (direction kept) for the probe.
            # We monkey-patch predict() to intercept the teach-slot s-vector before
            # the norm is taken — so the probe sees the full d-dim teach vectors,
            # not just 10 scalar magnitudes.
            def grab_features(xs):
                out = []
                for x in xs:
                    holder = {}
                    orig_predict = field.predict

                    def _probe_predict():
                        holder['tv'] = (
                            field.s[field.t_start:field.t_end]
                            .flatten().clone().cpu().numpy()
                        )
                        return orig_predict()

                    field.predict = _probe_predict
                    field.eval_predict(x, n_relax=args.n_relax)
                    field.predict = orig_predict
                    out.append(holder['tv'])
                return np.array(out)

            split_n = max(50, n // 2)
            Xtr = grab_features(te_x[:split_n]);  ytr = np.array(te_y[:split_n])
            Xte = grab_features(te_x[split_n:]);  yte = np.array(te_y[split_n:])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
            ceil = 100 * clf.score(sc.transform(Xte), yte)
            report['supervised_ceiling'] = round(ceil, 1)
            print(f"supervised probe ceiling = {ceil:.1f}%  "
                  f"(gap above calibrated = {ceil-cal_acc:.1f}% not yet communicated)",
                  flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ceiling probe skipped: {e}", flush=True)

    json.dump(report, open(args.out, 'w'), indent=2)
    print(f"\nwrote {args.out}\n{json.dumps(report, indent=2)}")


if __name__ == '__main__':
    main()
