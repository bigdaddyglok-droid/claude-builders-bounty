"""
UERF Forensics Comparison — diff multiple brain checkpoints side by side.

Reads JSON reports produced by `uerf_forensics.py --json` and renders a
comparison table + a comparison dashboard (population, crystallization funnel,
state distribution, per-class memory, golden/diversity scores).

USAGE
─────
  python uerf_forensics_compare.py \
      label1=report1.json label2=report2.json ... \
      --out comparison.png
"""
import sys
import os
import json
import argparse

import numpy as np


def load(specs):
    reports = []
    for spec in specs:
        if '=' in spec:
            label, path = spec.split('=', 1)
        else:
            label, path = os.path.basename(path := spec).replace('.json', ''), spec
        with open(path) as f:
            reports.append((label, json.load(f)))
    return reports


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def table(reports):
    rows = [
        ('experience_count', lambda r: r['identity']['experience_count']),
        ('alive',            lambda r: r['population']['alive']),
        ('capacity_util %',  lambda r: r['population']['capacity_utilization_pct']),
        ('interior',         lambda r: r['population']['interior']),
        ('locked',           lambda r: r['population']['locked']),
        ('lock_ratio %',     lambda r: r['population']['lock_ratio_pct']),
        ('births',           lambda r: r['identity']['births']),
        ('deaths',           lambda r: r['identity']['deaths']),
        ('crystallizations', lambda r: r['identity']['crystallization_events']),
        ('bonds',            lambda r: r['bonds']['total_bonds']),
        ('bond_density %',   lambda r: r['bonds']['density_pct']),
        ('mean_bond_str',    lambda r: r['bonds']['mean_strength']),
        ('golden_org %',     lambda r: r['emergent'].get('golden_organization_score', 0)),
        ('state_diversity %',lambda r: r['emergent'].get('state_diversity_pct', 0)),
        ('eff_rank',         lambda r: r.get('c_geometry', {}).get('effective_rank', 0)),
        ('golden_zone',      lambda r: r.get('crystallization', {}).get('n_golden_zone', 0)),
        ('valence>thr',      lambda r: r.get('crystallization', {}).get('n_valence_above_thr', 0)),
        ('crystallized_now', lambda r: r.get('crystallization', {}).get('n_crystallized_now', 0)),
    ]
    labels = [lab for lab, _ in reports]
    w0 = max(len(n) for n, _ in rows) + 1
    wc = max(14, max(len(l) for l in labels) + 2)
    line = f"  {'metric':<{w0}}" + ''.join(f"{l:>{wc}}" for l in labels)
    print(line)
    print("  " + "─" * (w0 + wc * len(labels)))
    for name, fn in rows:
        cells = ''.join(f"{fmt(fn(r)):>{wc}}" for _, r in reports)
        print(f"  {name:<{w0}}{cells}")


def figure(reports, out_path):
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
    labels = [lab for lab, _ in reports]
    palette = ['#58a6ff', '#f0883e', '#3fb950', '#bc8cff', '#f0c419']
    cols = [palette[i % len(palette)] for i in range(len(reports))]
    x = np.arange(len(reports))

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.24,
                  top=0.88, bottom=0.07, left=0.06, right=0.97)
    fig.suptitle("UERF FORENSICS — CHECKPOINT COMPARISON", fontsize=18,
                 fontweight='bold', y=0.955)
    fig.text(0.5, 0.91, "   vs   ".join(labels), ha='center',
             fontsize=11, color='#9da7b3')

    # (0,0) population stacked: sensory/teaching/interior-plastic/locked/dormant
    ax = fig.add_subplot(gs[0, 0])
    for i, (lab, r) in enumerate(reports):
        p = r['population']
        segs = [('sensory', p['sensory'], '#30363d'),
                ('teaching', p['teaching'], '#484f58'),
                ('plastic', p['plastic_interior'], '#58a6ff'),
                ('locked', p['locked'], '#f0c419'),
                ('dormant', p['dormant'], '#21262d')]
        bottom = 0
        for name, val, col in segs:
            ax.bar(i, val, bottom=bottom, color=col,
                   label=name if i == 0 else None, width=0.6, edgecolor='#0d1117')
            bottom += val
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=12)
    ax.set_ylabel('oscillators'); ax.set_title('Population Composition (of capacity)')
    ax.legend(fontsize=7, framealpha=0.2, loc='upper left')

    # (0,1) crystallization funnel grouped
    ax = fig.add_subplot(gs[0, 1])
    stages = ['alive', 'golden_zone', 'valence>thr', 'cryst_now', 'locked']
    keys = ['alive', 'n_golden_zone', 'n_valence_above_thr', 'n_crystallized_now', 'n_locked']
    width = 0.8 / len(reports)
    for i, (lab, r) in enumerate(reports):
        cr = r.get('crystallization', {})
        vals = [r['population']['alive'], cr.get('n_golden_zone', 0),
                cr.get('n_valence_above_thr', 0), cr.get('n_crystallized_now', 0),
                cr.get('n_locked', 0)]
        ax.bar(np.arange(len(stages)) + i * width, vals, width=width,
               color=cols[i], label=lab)
    ax.set_xticks(np.arange(len(stages)) + width * (len(reports)-1) / 2)
    ax.set_xticklabels(stages, fontsize=7, rotation=15)
    ax.set_ylabel('osc count'); ax.set_title('Crystallization Funnel')
    ax.legend(fontsize=7, framealpha=0.2)

    # (0,2) key scores grouped
    ax = fig.add_subplot(gs[0, 2])
    metrics = [('golden_org', lambda r: r['emergent'].get('golden_organization_score', 0)),
               ('diversity', lambda r: r['emergent'].get('state_diversity_pct', 0)),
               ('cap_util', lambda r: r['population']['capacity_utilization_pct']),
               ('lock_ratio', lambda r: r['population']['lock_ratio_pct']),
               ('bond_dens', lambda r: r['bonds']['density_pct'])]
    width = 0.8 / len(reports)
    for i, (lab, r) in enumerate(reports):
        vals = [fn(r) for _, fn in metrics]
        ax.bar(np.arange(len(metrics)) + i * width, vals, width=width, color=cols[i], label=lab)
    ax.set_xticks(np.arange(len(metrics)) + width * (len(reports)-1) / 2)
    ax.set_xticklabels([m for m, _ in metrics], fontsize=7, rotation=15)
    ax.set_ylabel('%'); ax.set_title('Key Scores (%)')
    ax.legend(fontsize=7, framealpha=0.2)

    # (1,0) births / bonds / crystallizations (log)
    ax = fig.add_subplot(gs[1, 0])
    metrics = [('births', lambda r: r['identity']['births']),
               ('crystalliz.', lambda r: r['identity']['crystallization_events']),
               ('bonds', lambda r: r['bonds']['total_bonds'])]
    width = 0.8 / len(reports)
    for i, (lab, r) in enumerate(reports):
        vals = [fn(r) for _, fn in metrics]
        ax.bar(np.arange(len(metrics)) + i * width, vals, width=width, color=cols[i], label=lab)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(metrics)) + width * (len(reports)-1) / 2)
    ax.set_xticklabels([m for m, _ in metrics], fontsize=8)
    ax.set_ylabel('count (log)'); ax.set_title('Growth & Memory (log scale)')
    ax.legend(fontsize=7, framealpha=0.2)

    # (1,1) state distribution overlay (top states)
    ax = fig.add_subplot(gs[1, 1])
    all_states = set()
    for _, r in reports:
        all_states.update(r['state_distribution'].keys())
    all_states = sorted(all_states)
    width = 0.8 / len(reports)
    for i, (lab, r) in enumerate(reports):
        sd = r['state_distribution']
        vals = [sd.get(s, {}).get('pct', 0) for s in all_states]
        ax.bar(np.arange(len(all_states)) + i * width, vals, width=width, color=cols[i], label=lab)
    ax.set_xticks(np.arange(len(all_states)) + width * (len(reports)-1) / 2)
    ax.set_xticklabels([s[:5] for s in all_states], fontsize=6, rotation=60)
    ax.set_ylabel('% interior'); ax.set_title('State Distribution (% interior)')
    ax.legend(fontsize=7, framealpha=0.2)

    # (1,2) per-class bond mass overlay
    ax = fig.add_subplot(gs[1, 2])
    for i, (lab, r) in enumerate(reports):
        active = r.get('active_classes', [])
        cm = r.get('class_memory', {})
        if active:
            mass = [cm.get(f'class_{c}', {}).get('bond_mass', 0) for c in active]
            ax.plot(active, mass, 'o-', color=cols[i], label=lab, ms=4)
    ax.set_xlabel('class'); ax.set_ylabel('bond mass')
    ax.set_title('Per-Class Memory Strength')
    ax.legend(fontsize=7, framealpha=0.2)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[viz] saved comparison → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', help="label=report.json specs")
    parser.add_argument('--out', default='comparison.png')
    args = parser.parse_args()
    reports = load(args.reports)
    print("\n" + "=" * 64)
    print(" UERF FORENSICS — CHECKPOINT COMPARISON")
    print("=" * 64 + "\n")
    table(reports)
    try:
        figure(reports, args.out)
    except Exception as e:
        import traceback
        print(f"[viz] failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
