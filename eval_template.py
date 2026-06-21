#!/usr/bin/env python3
"""
eval_template.py — Nearest-prototype template readout evaluation.

Fits per-class teach-vector templates from labeled training examples, then
evaluates accuracy using cosine-similarity nearest-prototype matching against
those templates. This uses the FULL 32-dim teach-slot state instead of a
scalar norm — the same information the supervised probe sees at 86.6%.

Compares four readout methods on the same checkpoint:
  raw         — existing scalar norm argmax (baseline)
  calib       — z-scored norm (domain-masked)
  template    — nearest-prototype cosine sim (no domain mask)
  tmpl_masked — nearest-prototype cosine sim (domain-masked to [lo:hi])

GPU only.
"""
import argparse, json, time
import torch
import numpy as np
import uerf_brain as U

DOMAINS = {
    'mnist': ('load_mnist',          0,   10),
    'cifar': ('load_cifar100',      10,  100),
    'ti':    ('load_tiny_imagenet', 110,  200),
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
        ys.append(int(lbl) + class_offset)
    return xs, ys


def raw_masked_acc(field, xs, ys, lo, hi, n_relax, n_calib_raw, seed, loader_name, device, class_offset):
    """Baseline: domain-masked calibrated norm readout."""
    import uerf_lifetime as L
    encoder = L.UniversalEncoder(device=device)
    proj = U.SensoryProjection(512, L.CONFIG['n_sensory'], seed=seed)
    proj.to(device)
    tr, _ = getattr(L, loader_name)()
    cal_xs = []
    for i in range(min(n_calib_raw, len(tr))):
        img, lbl = tr[i]
        feat = encoder.encode(img.unsqueeze(0))[0]
        x = proj(feat)
        x = x / x.norm().clamp(min=1e-6)
        cal_xs.append(x)

    raws = []
    t_start, t_end = field.t_start, field.t_end
    c_t = field.c[t_start:t_end]
    def _cap(xlist):
        out = []
        for x in xlist:
            holder = {}
            orig = field.predict
            def _p(_o=orig):
                tv = field.s[t_start:t_end]
                holder['r'] = tv.norm(dim=-1).clone().cpu().numpy()
                return _o()
            field.predict = _p
            field.eval_predict(x, n_relax=n_relax)
            field.predict = orig
            if 'r' in holder:
                out.append(holder['r'])
        return np.asarray(out)

    cal_r  = _cap(cal_xs)
    test_r = _cap(xs)
    mu, sd = cal_r.mean(0), cal_r.std(0).clip(min=1e-6)
    test_z = (test_r - mu) / sd
    sub    = test_z[:, lo:hi]
    pred   = sub.argmax(1) + lo
    return 100.0 * (pred == np.asarray(ys)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     required=True)
    ap.add_argument('--name',     default='ckpt')
    ap.add_argument('--domain',   default='mnist', choices=list(DOMAINS))
    ap.add_argument('--n_eval',   type=int, default=2000)
    ap.add_argument('--n_calib',  type=int, default=500)   # labeled examples for templates
    ap.add_argument('--n_relax',  type=int, default=8)
    ap.add_argument('--seed',     type=int, default=42)
    ap.add_argument('--out',      default='eval_template.json')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    field = U.UERFField.load_checkpoint(args.ckpt, device=dev)
    field._replay_disabled = True

    loader, lo, ncls = DOMAINS[args.domain]
    hi = lo + ncls
    print(f"[{args.name}/{args.domain}] slots [{lo}:{hi}]  "
          f"locked={int((field._class_locked & field.alive_mask).sum())}  "
          f"exp={int(field.experience_count)}", flush=True)

    # ── Labeled TRAIN split → fit templates ────────────────────────────────────
    print(f"[{args.name}] building {args.n_calib} labeled train features for templates...", flush=True)
    tr_xs, tr_ys = build_features(loader, args.n_calib, 'train', args.seed, dev, lo)
    print(f"[{args.name}] fitting teach templates ({len(tr_xs)} examples)...", flush=True)
    t0 = time.time()
    n_fit = field.fit_teach_templates(tr_xs, tr_ys, n_relax=args.n_relax,
                                      domain_lo=lo, domain_hi=hi)
    print(f"[{args.name}] templates fitted in {time.time()-t0:.0f}s  ({n_fit} classes)", flush=True)

    # ── TEST split → evaluate ──────────────────────────────────────────────────
    print(f"[{args.name}] building {args.n_eval} test features...", flush=True)
    te_xs, te_ys = build_features(loader, args.n_eval, 'test', args.seed, dev, lo)
    te_y = np.asarray(te_ys)

    # Template readout (full 32-dim, no domain mask)
    print(f"[{args.name}] evaluating template readout (unmasked)...", flush=True)
    t0 = time.time()
    tmpl_pred = []
    for x in te_xs:
        holder = {}
        orig = field.predict
        def _p(_o=orig):
            sims = field.predict_template()
            holder['sims'] = sims.clone().cpu().numpy() if sims is not None else None
            return _o()
        field.predict = _p
        field.eval_predict(x, n_relax=args.n_relax)
        field.predict = orig
        if holder.get('sims') is not None:
            tmpl_pred.append(holder['sims'].argmax())
        else:
            tmpl_pred.append(-1)
    tmpl_acc = 100.0 * (np.asarray(tmpl_pred) == te_y).mean()
    print(f"[{args.name}] template unmasked: {tmpl_acc:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    # Template readout (domain-masked)
    print(f"[{args.name}] evaluating template readout (domain-masked [{lo}:{hi}])...", flush=True)
    t0 = time.time()
    tmpl_m_pred = []
    for x in te_xs:
        holder = {}
        orig = field.predict
        def _p2(_o=orig):
            sims = field.predict_template(domain_lo=lo, domain_hi=hi)
            holder['sims'] = sims.clone().cpu().numpy() if sims is not None else None
            return _o()
        field.predict = _p2
        field.eval_predict(x, n_relax=args.n_relax)
        field.predict = orig
        if holder.get('sims') is not None:
            tmpl_m_pred.append(int(np.array(holder['sims'])[lo:hi].argmax()) + lo)
        else:
            tmpl_m_pred.append(-1)
    tmpl_m_acc = 100.0 * (np.asarray(tmpl_m_pred) == te_y).mean()
    print(f"[{args.name}] template masked:   {tmpl_m_acc:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    # Baseline: domain-masked calibrated norm
    print(f"[{args.name}] computing baseline masked-calib readout...", flush=True)
    base_acc = raw_masked_acc(field, te_xs, te_ys, lo, hi,
                              args.n_relax, 300, args.seed, loader, dev, lo)

    chance = 100.0 / ncls
    report = {
        'name': args.name, 'domain': args.domain, 'slots': [lo, hi],
        'n_eval': len(te_y), 'n_calib_templates': len(tr_xs), 'chance': round(chance, 2),
        'baseline_masked_calib': round(base_acc, 1),
        'template_unmasked':     round(tmpl_acc, 1),
        'template_masked':       round(tmpl_m_acc, 1),
        'probe_ceiling_mnist':   86.6,   # from eval_full run
    }
    json.dump(report, open(args.out, 'w'), indent=2)
    print(f"\n[{args.name}] TEMPLATE READOUT REPORT:\n{json.dumps(report, indent=2)}", flush=True)
    return report


if __name__ == '__main__':
    main()
