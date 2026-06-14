"""
UERF Brain Forensics — Full Introspection Tool  (v2)
═══════════════════════════════════════════════════════
Loads a brain checkpoint and produces a comprehensive forensic report
of its current state, plus a publication-quality visual dashboard.

SECTIONS
  1.  IDENTITY          — Per-brain config + lifetime counters
  2.  POPULATION        — State distribution, capacity utilization
  3.  PARAMETERS        — theta/S/f/N golden-anchor distributions
  4.  BONDS             — Network topology, hubs, degree distribution
  5.  CLASS MEMORY      — What each TRAINED class has stored
  6.  C-VECTOR GEOMETRY — Meaning-space rank, clustering, confusability
  7.  REPLAY BUFFER     — Stratified per-class replay occupancy
  8.  CRYSTALLIZATION   — Why locked/crystallized is what it is (NEW)
  9.  EMERGENT          — Self-organized properties
  10. DIAGNOSIS         — Findings + recommendations

VISUALS (--visualize) — a 4×3 dashboard:
  θ-S phase portrait · valence-vs-threshold · frequency 3-6-9
  c-vector PCA (state) · c-vector PCA (class) · singular spectrum
  bond heatmap · degree distribution · energy distribution
  state donut · per-class memory · crystallization funnel

USAGE
─────
  python uerf_forensics.py /path/to/main.pt
  python uerf_forensics.py /path/to/main.pt --visualize
  python uerf_forensics.py /path/to/main.pt --visualize --json out.json
  python uerf_forensics.py /path/to/main.pt --out-dir ./forensics_out
"""

import sys
import os
import argparse
import json
import math
from collections import Counter

import torch
import numpy as np


# ── small formatting helpers ────────────────────────────────────────────────
def _hr(title, ch='='):
    return f"\n{ch*64}\n {title}\n{ch*64}"


def _stat(t):
    if t is None or len(t) == 0:
        return dict(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0)
    t = t.float()
    return dict(
        mean=float(t.mean()), std=float(t.std()),
        min=float(t.min()), max=float(t.max()),
        median=float(t.median()),
    )


def _bar(frac, width=24):
    frac = max(0.0, min(1.0, float(frac)))
    n = int(round(frac * width))
    return '█' * n + '░' * (width - n)


# ═══════════════════════════════════════════════════════════════════════════
def analyze(field, STATE_NAMES, THETA_G, S_PHI, ckpt_path):
    report = {}

    # ── 1. IDENTITY ─────────────────────────────────────────────────────────
    print(_hr("1. IDENTITY"))
    identity = {
        'n_max': int(field.n_max),
        'n_classes': int(field.n_classes),
        'input_dim': int(field.input_dim) if field.input_dim else None,
        'd': int(field.d),
        'experience_count': int(field.experience_count),
        'births': int(field.births),
        'deaths': int(field.deaths),
        'crystallization_events': int(field.crystallization_events),
        't_start': int(field.t_start) if field.t_start is not None else None,
        't_end': int(field.t_end) if field.t_end is not None else None,
        'file_mb': round(os.path.getsize(ckpt_path) / 1e6, 1),
    }
    for k, v in identity.items():
        print(f"  {k:28s} = {v}")
    report['identity'] = identity

    # ── 2. POPULATION ───────────────────────────────────────────────────────
    print(_hr("2. POPULATION & STATE DISTRIBUTION"))
    alive = field.alive_mask
    sensory = getattr(field, '_is_sensory', torch.zeros_like(alive))
    teaching = getattr(field, '_is_teaching', torch.zeros_like(alive))
    interior = alive & ~sensory & ~teaching
    locked = getattr(field, '_class_locked', torch.zeros_like(alive))
    n_alive = int(alive.sum())
    n_int = int(interior.sum())

    pop = {
        'total_capacity': int(field.n_max),
        'alive': n_alive,
        'dormant': int((~alive).sum()),
        'sensory': int(sensory.sum()),
        'teaching': int(teaching.sum()),
        'interior': n_int,
        'locked': int(locked.sum()),
        'plastic_interior': int((interior & ~locked).sum()),
        'capacity_utilization_pct': 100 * n_alive / field.n_max,
        'lock_ratio_pct': 100 * int(locked.sum()) / max(n_int, 1),
    }
    for k, v in pop.items():
        line = f"  {k:28s} = {v:.1f}" if isinstance(v, float) else f"  {k:28s} = {v}"
        print(line)
    print(f"\n  capacity  {_bar(n_alive / field.n_max)}  "
          f"{n_alive}/{field.n_max} alive")
    print(f"  locked    {_bar(int(locked.sum()) / max(n_int, 1))}  "
          f"{int(locked.sum())}/{n_int} interior locked")
    report['population'] = pop

    print(f"\n  STATE DISTRIBUTION (interior only):")
    state_counts = Counter(field.state_id[interior].tolist())
    state_dist = {}
    for sid, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        name = STATE_NAMES.get(sid, f'STATE_{sid}')
        pct = 100 * count / max(n_int, 1)
        print(f"    {name:15s} ({sid:2d}): {count:4d}  {_bar(pct/100, 16)} {pct:5.1f}%")
        state_dist[name] = {'id': int(sid), 'count': int(count), 'pct': pct}
    report['state_distribution'] = state_dist

    # ── 3. PARAMETERS ───────────────────────────────────────────────────────
    print(_hr("3. PARAMETER DISTRIBUTIONS (alive oscillators)"))
    print(f"  Golden anchors: THETA_G={THETA_G:.4f}  S_PHI={S_PHI:.4f}")
    params = {}
    for attr in ['theta', 'S', 'f', 'N', 'a0', 'age']:
        if hasattr(field, attr):
            s = _stat(getattr(field, attr)[alive])
            params[attr] = s
            print(f"  {attr:6s} mean={s['mean']:+.4f}  std={s['std']:.4f}  "
                  f"range=[{s['min']:+.3f}, {s['max']:+.3f}]")

    near_theta = int(((field.theta[alive] - THETA_G).abs() < 0.1).sum())
    near_S = int(((field.S[alive] - S_PHI).abs() < 0.1).sum())
    near_both = int((((field.theta[alive] - THETA_G).abs() < 0.1) &
                     ((field.S[alive] - S_PHI).abs() < 0.1)).sum())
    params['near_golden_theta_pct'] = 100 * near_theta / max(n_alive, 1)
    params['near_golden_S_pct'] = 100 * near_S / max(n_alive, 1)
    params['near_golden_both_pct'] = 100 * near_both / max(n_alive, 1)
    print(f"\n  Near golden θ  (|Δ|<0.1): {near_theta:4d}/{n_alive}  "
          f"({params['near_golden_theta_pct']:.1f}%)")
    print(f"  Near golden S  (|Δ|<0.1): {near_S:4d}/{n_alive}  "
          f"({params['near_golden_S_pct']:.1f}%)")
    print(f"  Near BOTH (golden-zone):  {near_both:4d}/{n_alive}  "
          f"({params['near_golden_both_pct']:.1f}%)  ← crystallization-eligible")
    report['parameters'] = params

    print(f"\n  FREQUENCY 3-6-9 harmonics:")
    f_vals = field.f[alive]
    for tf in [3.0, 6.0, 9.0]:
        n_near = int(((f_vals - tf).abs() < 0.5).sum())
        print(f"    f≈{tf:.0f} (|Δ|<0.5): {n_near:4d}  ({100*n_near/max(n_alive,1):.1f}%)")

    # ── 4. BONDS ────────────────────────────────────────────────────────────
    print(_hr("4. BOND TOPOLOGY"))
    C_mask = field.C_mask
    bstr = field.C.norm(dim=(-2, -1))     # (n,n)
    bonds = {
        'total_bonds': int(C_mask.sum()),
        'bonds_per_alive': float(C_mask.sum()) / max(n_alive, 1),
        'density_pct': 100 * float(C_mask.sum()) / max(n_alive * n_alive, 1),
        'mean_strength': float(bstr[C_mask].mean()) if C_mask.any() else 0.0,
        'max_strength': float(bstr.max()),
        'total_bond_mass': float(bstr.sum()),
    }
    for k, v in bonds.items():
        print(f"  {k:28s} = {v:.3f}" if isinstance(v, float) else f"  {k:28s} = {v}")

    in_strength = bstr.sum(dim=0)
    out_strength = bstr.sum(dim=1)
    in_deg = C_mask.sum(dim=0).float()
    out_deg = C_mask.sum(dim=1).float()
    report['bonds'] = bonds

    int_idx = interior.nonzero(as_tuple=True)[0]
    if len(int_idx) > 0:
        top_in = in_strength[int_idx].topk(min(8, len(int_idx)))
        print(f"\n  TOP ATTRACTOR HUBS (highest incoming bond mass):")
        hubs = []
        for k in range(len(top_in.indices)):
            idx = int(int_idx[top_in.indices[k]])
            mass = float(top_in.values[k])
            sid = int(field.state_id[idx])
            lc = int(field._locked_class[idx]) if hasattr(field, '_locked_class') else -1
            tag = f"LOCKED→class {lc}" if locked[idx] else "free"
            print(f"    osc {idx:4d}: in={mass:7.2f}  {STATE_NAMES.get(sid,'?'):13s}  {tag}")
            hubs.append({'osc': idx, 'in_mass': mass, 'state': sid, 'locked_class': lc})
        report['hubs'] = hubs

    # ── 5. CLASS MEMORY ─────────────────────────────────────────────────────
    print(_hr("5. CLASS MEMORY (trained classes only)"))
    class_memory = {}
    active_classes = []
    if getattr(field, '_teach_bond_count', None) is not None and field.t_start is not None:
        sensor_count = int(sensory.sum())
        for c in range(field.n_classes):
            t_idx = field.t_start + c
            count = int(field._teach_bond_count[c])
            mass = float(bstr[t_idx, :sensor_count].sum()) if sensor_count > 0 \
                else float(bstr[t_idx].sum())
            nlock = int((field._locked_class == c).sum()) if hasattr(field, '_locked_class') else 0
            if count > 0 or nlock > 0 or mass > 1e-4:
                active_classes.append(c)
            class_memory[f'class_{c}'] = {
                'training_count': count, 'bond_mass': mass, 'locked_osc_count': nlock,
            }
        print(f"  Active (trained) classes: {active_classes}")
        print(f"\n  {'class':>5}  {'trained':>8}  {'bond_mass':>10}  {'locked':>6}")
        mm = [class_memory[f'class_{c}']['bond_mass'] for c in active_classes] or [1.0]
        mmax = max(mm) or 1.0
        for c in active_classes:
            m = class_memory[f'class_{c}']
            print(f"  {c:>5}  {m['training_count']:>8}  {m['bond_mass']:>10.3f}  "
                  f"{m['locked_osc_count']:>6}  {_bar(m['bond_mass']/mmax, 16)}")
    report['class_memory'] = class_memory
    report['active_classes'] = active_classes

    # ── 6. C-VECTOR GEOMETRY ────────────────────────────────────────────────
    print(_hr("6. C-VECTOR GEOMETRY (meaning space)"))
    c_int = field.c[interior]
    cg = {}
    if len(c_int) > 1:
        gram = c_int @ c_int.T
        n = gram.shape[0]
        off = gram[~torch.eye(n, dtype=torch.bool)]
        cg['cosine_stats'] = _stat(off)
        print(f"  Pairwise cosine (interior): mean={cg['cosine_stats']['mean']:+.4f}  "
              f"std={cg['cosine_stats']['std']:.4f}")
        U, Sv, _ = torch.linalg.svd(c_int, full_matrices=False)
        thr = Sv[0] * 0.01
        eff_rank = int((Sv > thr).sum())
        cg['effective_rank'] = eff_rank
        cg['singular_values'] = Sv[:min(len(Sv), field.d)].tolist()
        print(f"  Effective rank: {eff_rank} / {min(len(c_int), field.d)}  "
              f"(independent meaning-directions)")
        print(f"  Top singular values: {[round(s,3) for s in Sv[:6].tolist()]}")
        if field.t_start is not None:
            teach_c = field.c[field.t_start:field.t_end]
            i2t = c_int @ teach_c.T
            cg['mean_max_teach_align'] = float(i2t.abs().max(dim=1).values.mean())
            print(f"  Interior→Teach mean max-alignment: {cg['mean_max_teach_align']:+.3f}")
        # class-confusability: cosine between trained teach c-vectors
        if active_classes and field.t_start is not None:
            tc = field.c[[field.t_start + c for c in active_classes]]
            tc = tc / tc.norm(dim=1, keepdim=True).clamp(min=1e-6)
            conf = tc @ tc.T
            m = conf.clone()
            m.fill_diagonal_(-1)
            mi = int(m.argmax())
            a, b = mi // m.shape[1], mi % m.shape[1]
            print(f"  Most confusable class pair: "
                  f"{active_classes[a]} ↔ {active_classes[b]}  cos={float(m[a,b]):+.3f}")
            cg['most_confusable'] = [active_classes[a], active_classes[b], float(m[a, b])]
    report['c_geometry'] = cg

    # ── 7. REPLAY BUFFER ────────────────────────────────────────────────────
    print(_hr("7. REPLAY BUFFER"))
    replay = {}
    if getattr(field, '_replay_per_class', None):
        total = 0
        for c, buf in field._replay_per_class.items():
            replay[str(c)] = len(buf)
            total += len(buf)
        print(f"  Total buffered: {total}")
        for c in sorted(replay, key=lambda k: int(k)):
            print(f"    class {c}: {replay[c]}")
    else:
        print("  No replay buffer found (or empty).")
    report['replay'] = replay

    # ── 8. CRYSTALLIZATION DIAGNOSIS (NEW) ──────────────────────────────────
    print(_hr("8. CRYSTALLIZATION / LOCK DIAGNOSIS"))
    cryst = {}
    thr = float(getattr(field, '_crystallization_threshold', 0.25))
    cryst['threshold'] = thr
    if hasattr(field, '_valence_accumulator'):
        va = field._valence_accumulator
        va_alive = va[alive]
        cryst['valence_stats'] = _stat(va_alive)
        golden_zone = (((field.theta - THETA_G).abs() < 0.10) &
                       ((field.S - S_PHI).abs() < 0.10) & alive)
        valence_ok = (va > thr) & alive
        crystallized = golden_zone & valence_ok
        n_golden = int(golden_zone.sum())
        n_val = int(valence_ok.sum())
        n_cryst = int(crystallized.sum())
        cryst['n_golden_zone'] = n_golden
        cryst['n_valence_above_thr'] = n_val
        cryst['n_crystallized_now'] = n_cryst
        cryst['n_locked'] = int(locked.sum())
        print(f"  Crystallization requires BOTH:")
        print(f"    (a) golden-zone  (|θ-θ_G|<0.1 ∧ |S-S_φ|<0.1): {n_golden:4d} osc")
        print(f"    (b) valence > threshold ({thr:.2f})           : {n_val:4d} osc")
        print(f"    → crystallized NOW (a∧b)                      : {n_cryst:4d} osc")
        print(f"    → class-LOCKED (consolidate_class froze these): {int(locked.sum()):4d} osc")
        print(f"\n  Valence accumulator (alive): mean={cryst['valence_stats']['mean']:+.4f}  "
              f"max={cryst['valence_stats']['max']:+.4f}  thr={thr:.2f}")
        # why-zero reasoning
        if n_cryst == 0:
            if n_golden == 0:
                why = ("No alive osc sit in the golden zone — dynamics never drove "
                       "θ,S to the φ attractor, so physics-crystallization cannot fire.")
            elif n_val == 0:
                why = (f"Osc reach the golden zone ({n_golden}) but none exceed the "
                       f"valence threshold ({thr:.2f}); max valence is "
                       f"{cryst['valence_stats']['max']:.3f}. Memory is preserved by "
                       f"class-LOCKING instead (consolidate_class), not by earned valence.")
            else:
                why = ("Golden-zone and high-valence sets exist but don't overlap on the "
                       "same oscillators.")
        else:
            why = f"{n_cryst} osc satisfy both criteria — physics-crystallization is active."
        print(f"\n  WHY: {why}")
        cryst['why'] = why
        print(f"\n  NOTE: live training logs print locked=0 because consolidate_class() "
              f"\n        only runs AFTER a phase finishes. This saved checkpoint shows "
              f"\n        locked={int(locked.sum())}, crystallization_events="
              f"{int(field.crystallization_events)}.")
    report['crystallization'] = cryst

    # ── 9. EMERGENT ─────────────────────────────────────────────────────────
    print(_hr("9. EMERGENT PROPERTIES"))
    emergent = {}
    golden_org = (params['near_golden_theta_pct'] + params['near_golden_S_pct']) / 2
    emergent['golden_organization_score'] = golden_org
    print(f"  Golden-anchor organization: {golden_org:.1f}%  "
          f"({'STRONG' if golden_org>30 else 'MODERATE' if golden_org>10 else 'WEAK'})")
    if state_dist:
        probs = np.array([s['count'] for s in state_dist.values()], float)
        probs = probs / probs.sum()
        ent = -np.sum(probs * np.log(probs + 1e-12))
        div = 100 * ent / max(math.log(len(state_dist)), 1e-6)
        emergent['state_diversity_pct'] = div
        print(f"  State diversity: {div:.1f}%  "
              f"({'HIGH' if div>70 else 'MODERATE' if div>40 else 'LOW'})")
    plastic_pct = 100 * pop['plastic_interior'] / max(n_int, 1)
    dormant_pct = 100 * pop['dormant'] / pop['total_capacity']
    emergent['plastic_pct'] = plastic_pct
    emergent['dormant_pct'] = dormant_pct
    print(f"  Plastic interior: {pop['plastic_interior']} ({plastic_pct:.1f}%)")
    print(f"  Dormant capacity: {pop['dormant']} ({dormant_pct:.1f}%)")
    report['emergent'] = emergent

    # ── 10. DIAGNOSIS ───────────────────────────────────────────────────────
    print(_hr("10. DIAGNOSIS"))
    dx, rec = [], []
    if pop['dormant'] == 0:
        dx.append("Field at full capacity (no dormant slots).")
        rec.append("Enable grow_capacity to let the field expand.")
    if pop['lock_ratio_pct'] > 50:
        dx.append(f"High lock ratio ({pop['lock_ratio_pct']:.0f}%) — plastic substrate shrinking.")
        rec.append("Grow capacity or reduce n_lock_per_cls.")
    if state_dist:
        top = max(state_dist.items(), key=lambda kv: kv[1]['pct'])
        if top[1]['pct'] > 70:
            dx.append(f"Single-state dominance: {top[0]} at {top[1]['pct']:.0f}%.")
            rec.append(f"Verify {top[0]} concentration is specialization, not monoculture.")
    if golden_org < 10:
        dx.append(f"Low golden-anchor alignment ({golden_org:.1f}%).")
        rec.append("φ-attractors weakly engaged — physics-crystallization will stay rare.")
    if identity['deaths'] == 0 and identity['births'] > 0:
        dx.append(f"Growth without pruning: {identity['births']} births, 0 deaths.")
        rec.append("Aging/cull not removing osc — check cull thresholds if capacity tightens.")
    if identity['crystallization_events'] > 0:
        dx.append(f"Consolidation occurred: {identity['crystallization_events']} crystallization events, "
                  f"{pop['locked']} osc locked.")
    if not dx:
        dx.append("Brain healthy and balanced.")
        rec.append("Continue training.")
    print("  FINDINGS:")
    for d in dx:
        print(f"    • {d}")
    print("  RECOMMENDATIONS:")
    for r in rec:
        print(f"    • {r}")
    report['diagnosis'] = dx
    report['recommendations'] = rec

    # cache tensors the visualizer needs
    report['_viz'] = dict(
        alive=alive, interior=interior, locked=locked, sensory=sensory,
        in_strength=in_strength, out_strength=out_strength,
        in_deg=in_deg, out_deg=out_deg,
    )
    return report


# ═══════════════════════════════════════════════════════════════════════════
def visualize(field, report, ckpt_path, STATE_NAMES, THETA_G, S_PHI, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'savefig.facecolor': '#0d1117', 'text.color': '#e6edf3',
        'axes.labelcolor': '#e6edf3', 'xtick.color': '#9da7b3',
        'ytick.color': '#9da7b3', 'axes.edgecolor': '#30363d',
        'axes.titlecolor': '#e6edf3', 'font.size': 9,
    })
    ACCENT, GOLD = '#58a6ff', '#f0c419'

    alive = report['_viz']['alive']
    interior = report['_viz']['interior']
    locked = report['_viz']['locked']
    n_alive = int(alive.sum())

    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.26,
                  top=0.90, bottom=0.05, left=0.06, right=0.97)
    ident = report['identity']
    pop = report['population']
    fig.suptitle(
        f"UERF BRAIN FORENSICS   ·   {os.path.basename(ckpt_path)}",
        fontsize=18, fontweight='bold', color='#e6edf3', y=0.965)
    fig.text(0.5, 0.925,
             f"exp={ident['experience_count']:,}   alive={pop['alive']}/{pop['total_capacity']}"
             f"   locked={pop['locked']}   crystallizations={ident['crystallization_events']}"
             f"   births={ident['births']}   bonds={report['bonds']['total_bonds']:,}"
             f"   d={ident['d']}   classes={ident['n_classes']}",
             ha='center', fontsize=11, color='#9da7b3')

    th = field.theta[alive].numpy()
    Sv = field.S[alive].numpy()
    fv = field.f[alive].numpy()
    lk = locked[alive].numpy()

    # (0,0) θ-S phase portrait
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(th[~lk], Sv[~lk], s=14, c=ACCENT, alpha=0.6, label='plastic', edgecolors='none')
    ax.scatter(th[lk], Sv[lk], s=34, c=GOLD, alpha=0.95, label='locked', edgecolors='black', linewidths=0.4)
    ax.axvline(THETA_G, color=GOLD, ls='--', lw=1, alpha=0.7)
    ax.axhline(S_PHI, color=GOLD, ls='--', lw=1, alpha=0.7)
    ax.add_patch(plt.Rectangle((THETA_G-0.1, S_PHI-0.1), 0.2, 0.2,
                 fill=False, edgecolor=GOLD, lw=1.2, ls=':'))
    ax.scatter([THETA_G], [S_PHI], marker='*', s=240, c=GOLD, edgecolors='black', zorder=5)
    ax.set_xlabel('θ'); ax.set_ylabel('S'); ax.set_title('θ–S Phase Portrait (φ-attractor)')
    ax.legend(loc='upper right', framealpha=0.2, fontsize=8)

    # (0,1) valence vs threshold
    ax = fig.add_subplot(gs[0, 1])
    if hasattr(field, '_valence_accumulator'):
        va = field._valence_accumulator[alive].numpy()
        thr = report['crystallization'].get('threshold', 0.25)
        ax.hist(va, bins=40, color=ACCENT, alpha=0.8)
        ax.axvline(thr, color='#f85149', lw=2, label=f'threshold {thr:.2f}')
        ax.legend(fontsize=8, framealpha=0.2)
        ax.set_xlabel('valence accumulator'); ax.set_ylabel('count')
    ax.set_title('Valence vs Crystallization Threshold')

    # (0,2) frequency 3-6-9
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(fv, bins=40, color='#bc8cff', alpha=0.85)
    for t in (3, 6, 9):
        ax.axvline(t, color=GOLD, ls='--', lw=1, alpha=0.7)
    ax.set_xlabel('frequency f'); ax.set_ylabel('count')
    ax.set_title('Frequency Spectrum (3-6-9 marked)')

    # PCA of interior c-vectors
    c_int = field.c[interior].numpy()
    proj = None
    if len(c_int) > 2:
        X = c_int - c_int.mean(0, keepdims=True)
        U, S_, Vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ Vt[:2].T

    # (1,0) PCA colored by state
    ax = fig.add_subplot(gs[1, 0])
    if proj is not None:
        sids = field.state_id[interior].numpy()
        uniq = sorted(set(sids.tolist()))
        cmap = plt.cm.tab10
        for i, sid in enumerate(uniq):
            m = sids == sid
            ax.scatter(proj[m, 0], proj[m, 1], s=16, color=cmap(i % 10),
                       alpha=0.75, label=STATE_NAMES.get(sid, f'S{sid}')[:8], edgecolors='none')
        ax.legend(fontsize=6, framealpha=0.2, ncol=2, loc='best')
    ax.set_title('c-vector PCA — colored by STATE')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

    # (1,1) PCA colored by class affinity
    ax = fig.add_subplot(gs[1, 1])
    if proj is not None and field.t_start is not None:
        teach_c = field.c[field.t_start:field.t_end]
        i2t = (field.c[interior] @ teach_c.T)
        active = report.get('active_classes', [])
        if active:
            sub = i2t[:, active]
            aff = sub.argmax(1).numpy()
            sc = ax.scatter(proj[:, 0], proj[:, 1], s=18, c=aff, cmap='turbo',
                            alpha=0.8, edgecolors='none')
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('argmax class idx', fontsize=7)
    ax.set_title('c-vector PCA — colored by CLASS affinity')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

    # (1,2) singular spectrum
    ax = fig.add_subplot(gs[1, 2])
    sv = report.get('c_geometry', {}).get('singular_values', [])
    if sv:
        ax.plot(range(1, len(sv)+1), sv, 'o-', color=ACCENT, ms=4)
        er = report['c_geometry'].get('effective_rank', 0)
        ax.axvline(er, color='#f85149', ls='--', lw=1.5, label=f'eff. rank {er}')
        ax.legend(fontsize=8, framealpha=0.2)
    ax.set_xlabel('component'); ax.set_ylabel('singular value')
    ax.set_title('Meaning-space Singular Spectrum')

    # (2,0) bond heatmap (alive×alive, ordered by state)
    ax = fig.add_subplot(gs[2, 0])
    bstr = field.C.norm(dim=(-2, -1)).numpy()
    ai = alive.nonzero(as_tuple=True)[0].numpy()
    order = ai[np.argsort(field.state_id[alive].numpy())]
    if len(order) > 120:
        order = order[np.linspace(0, len(order)-1, 120).astype(int)]
    sub = bstr[np.ix_(order, order)]
    im = ax.imshow(sub, cmap='inferno', aspect='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'Bond Strength Matrix ({len(order)}² ordered by state)')

    # (2,1) degree distribution
    ax = fig.add_subplot(gs[2, 1])
    ind = report['_viz']['in_deg'][alive].numpy()
    outd = report['_viz']['out_deg'][alive].numpy()
    bins = np.linspace(0, max(ind.max(), outd.max(), 1), 30)
    ax.hist(ind, bins=bins, alpha=0.6, color=ACCENT, label='in-degree')
    ax.hist(outd, bins=bins, alpha=0.6, color='#f0883e', label='out-degree')
    ax.set_yscale('log')
    ax.legend(fontsize=8, framealpha=0.2)
    ax.set_xlabel('degree'); ax.set_ylabel('count (log)')
    ax.set_title('Bond Degree Distribution')

    # (2,2) energy distribution
    ax = fig.add_subplot(gs[2, 2])
    try:
        E = field.energy()[alive].numpy()
        ax.hist(E, bins=40, color='#3fb950', alpha=0.85)
        ax.set_xlabel('oscillator energy'); ax.set_ylabel('count')
    except Exception:
        ax.text(0.5, 0.5, 'energy() unavailable', ha='center', va='center')
    ax.set_title('Energy Distribution')

    # (3,0) state donut
    ax = fig.add_subplot(gs[3, 0])
    sd = report['state_distribution']
    if sd:
        labels = list(sd.keys())
        sizes = [sd[l]['count'] for l in labels]
        wedges, _, _ = ax.pie(sizes, labels=[l[:8] for l in labels], autopct='%1.0f%%',
                              startangle=90, wedgeprops=dict(width=0.42, edgecolor='#0d1117'),
                              textprops=dict(fontsize=7), pctdistance=0.78)
    ax.set_title('Interior State Distribution')

    # (3,1) per-class memory
    ax = fig.add_subplot(gs[3, 1])
    active = report.get('active_classes', [])
    cm = report.get('class_memory', {})
    if active:
        mass = [cm[f'class_{c}']['bond_mass'] for c in active]
        lockn = [cm[f'class_{c}']['locked_osc_count'] for c in active]
        x = np.arange(len(active))
        ax.bar(x, mass, color=ACCENT, alpha=0.85, label='bond mass')
        ax2 = ax.twinx()
        ax2.plot(x, lockn, 'o-', color=GOLD, label='locked osc', ms=5)
        ax2.set_ylabel('locked osc', color=GOLD)
        ax2.tick_params(axis='y', colors=GOLD)
        ax.set_xticks(x); ax.set_xticklabels([str(c) for c in active], fontsize=7)
        ax.set_xlabel('class'); ax.set_ylabel('bond mass', color=ACCENT)
    ax.set_title('Per-Class Memory Strength')

    # (3,2) crystallization funnel
    ax = fig.add_subplot(gs[3, 2])
    cr = report.get('crystallization', {})
    stages = ['alive', 'golden\nzone', 'valence\n>thr', 'crystal\nNOW', 'class\nlocked']
    vals = [n_alive, cr.get('n_golden_zone', 0), cr.get('n_valence_above_thr', 0),
            cr.get('n_crystallized_now', 0), cr.get('n_locked', 0)]
    colors = [ACCENT, '#8957e5', '#bc8cff', '#f0883e', GOLD]
    ax.barh(range(len(stages)), vals, color=colors, alpha=0.9)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f' {v}', va='center', fontsize=9, color='#e6edf3')
    ax.set_xlabel('oscillator count')
    ax.set_title('Crystallization Funnel')

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_png = os.path.join(out_dir, f'{base}_forensics.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[viz] saved dashboard → {out_png}")
    return out_png


# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help="Path to brain checkpoint (.pt)")
    parser.add_argument('--visualize', action='store_true', help="Save the visual dashboard")
    parser.add_argument('--json', type=str, default=None, help="Save full report as JSON")
    parser.add_argument('--out-dir', type=str, default='.', help="Output dir for PNG/JSON")
    parser.add_argument('--brain-module', default='uerf_brain')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.ckpt)))
    sys.path.insert(0, '.')
    sys.path.insert(0, '/uerf-output')
    sys.path.insert(0, '/kaggle/working')
    try:
        brain_mod = __import__(args.brain_module)
    except ImportError as e:
        print(f"ERROR: Could not import {args.brain_module}: {e}")
        sys.exit(1)

    UERFField = brain_mod.UERFField
    STATE_NAMES = brain_mod.STATE_NAMES
    THETA_G = brain_mod.THETA_G
    S_PHI = brain_mod.S_PHI

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              UERF BRAIN FORENSICS REPORT  (v2)               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Checkpoint: {args.ckpt}  ({os.path.getsize(args.ckpt)/1e6:.1f} MB)")
    print("Loading brain (CPU)...")
    field = UERFField.load_checkpoint(args.ckpt, device='cpu')

    report = analyze(field, STATE_NAMES, THETA_G, S_PHI, args.ckpt)

    if args.visualize:
        try:
            visualize(field, report, args.ckpt, STATE_NAMES, THETA_G, S_PHI, args.out_dir)
        except Exception as e:
            import traceback
            print(f"[viz] failed: {e}")
            traceback.print_exc()

    if args.json:
        def ser(o):
            if isinstance(o, torch.Tensor):
                return o.tolist()
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, dict):
                return {str(k): ser(v) for k, v in o.items() if k != '_viz'}
            if isinstance(o, list):
                return [ser(x) for x in o]
            return o
        out_json = args.json if os.path.isabs(args.json) else os.path.join(args.out_dir, args.json)
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(ser(report), f, indent=2, default=str)
        print(f"[saved] report → {out_json}")

    print(_hr("END REPORT"))


if __name__ == '__main__':
    main()
