"""
UERF BRAIN — Framework-Faithful Implementation
====================================================================
Patent pending #63/940,078 — Black

18 physical states are defined (S_CLASSICAL=0 … S_CONSCIOUSNESS=17): the 14
canonical UERF regimes, plus HoloAdS (AdS-boundary variant of Holographic) and
3 speculative extensions (Consciousness, Multiversal, Retrocausal).
CANDIDATE_STATES routes oscillators through 17 of them. Per-state α-ranges
match the framework document exactly.

THE CONTRACT (what makes this faithful, not decorative)
-------------------------------------------------------
  • Per oscillator: a d-dim state vector s.
        E = ||s||²              ← energy (the 6 equations govern this)
        s / ||s||               ← direction (carries the pattern/cognition)
    Physics drives magnitude. Direction carries the payload.

  • Eq 2/4 (Feedback / Master Discrete) govern E[n+1] = α·E + β·P_in + cross_flow.
  • Eq 3 (Preservation Factor α) = the universal core
        α₀·exp(-|θ-θ_g|/σ_θ)·exp(-(S-S_φ)²/2σ_S²)·(1+κ·Σ_{3,6,9})
    times a per-state modifier (state_modifier), clamped to the state's α-range.
  • Eq 5 (Master Continuous) dissipation = Λ₀·M_state / (F_φ·F_vortex·F_369),
    where M_state is the per-state loss law (state_dissipation_modifier):
    Holographic exp(−S_ent)≈0, Thermal exp(E_a/kT), Harmonic Q-factor,
    Phantom |1+w|, Fractal λ^D_f, Relativistic 1/γ, …
  • Eq 6 (Valence) V(s) = α⟨s,R⟩ − λ‖s−I‖² — the per-state routing objective.
  • Routing between states is emergent: argmax valence over states.

STATE → NEURAL FUNCTION
  Classical    baseline interior processing           (α 0.70–0.90)
  CubitClassic discrete-space baseline                (α 0.90–0.96)
  Quantum      superposition (parallel representations)(α 0.94–0.98)
  Qubit        discrete superposition                 (α 0.97–0.995)
  Phantom      growth regions (energy gain, α>1)      (α 1.00–1.05)
  Temporal     maturation (preservation earned by time)(α 0.80–0.93)
  Relativistic time-dilation (protects fast-changing) (α 0.85–0.94)
  Harmonic     pattern matching / resonance           (α 0.93–0.98)
  Vacuum       stable standby reservoir               (α 0.96–0.99)
  Fractal      multi-scale coupling                   (α 0.95–0.99)
  Retrocausal  predictive future-feedback             (α 0.94–0.98)
  Elemental    coupled-force gating                   (α 0.88–0.95)
  Holographic  long-term memory (near-perfect hold)   (α 0.98–0.998)
  HoloAdS      AdS/CFT boundary compression           (α 0.98–0.998)
  Thermal      entropy / exploration (annealing)      (α 0.75–0.92)
  Multiversal  parallel hypotheses                    (α 0.94–0.98)
  Toroidal     input / closed-loop flow               (α 0.92–0.97)
  Consciousness attention / salience gate             (α 0.90–0.96)
"""
import os, math, time, json
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — golden anchors from the framework
# ═══════════════════════════════════════════════════════════════════════════════
def _pick_device():
    """CUDA must actually work, not just be present — a GPU whose compute
    capability the installed torch wasn't built for (e.g. P100/sm_60 on
    modern wheels) passes is_available() but fails on the first kernel."""
    if torch.cuda.is_available():
        try:
            (torch.randn(4, device='cuda') @ torch.randn(4, 4, device='cuda'))
            torch.cuda.synchronize()
            return torch.device('cuda')
        except Exception as e:
            print(f"[device] CUDA present but unusable ({type(e).__name__}); using CPU")
    return torch.device('cpu')

DEVICE = _pick_device()
PHI       = 1.6180339887498948482          # golden ratio  (S_φ)
PHI_INV   = 0.6180339887498948482
THETA_G   = 2.3999632297286533             # golden angle (radians) ≈ 137.5°
S_PHI     = PHI                            # scale anchor (document's S_φ)
SIGMA_TH  = 0.55                           # σ_θ
SIGMA_S   = 0.55                           # σ_S
SIGMA_F   = 0.55                           # σ_f
SIGMA_N   = 2.1                            # σ_N
KAPPA     = 0.15                           # κ  (3-6-9 coupling)
HARM      = (3.0, 6.0, 9.0)               # the 3-6-9 family
F0        = 6.0                            # f_0 reference (mid of 3-6-9)
LAMBDA0   = 1.0                            # Λ_0 base dissipation (Eq 5)

# ═══════════════════════════════════════════════════════════════════════════════
# Eq 3 — PRESERVATION FACTOR.  Universal core + per-state modifier.
# ═══════════════════════════════════════════════════════════════════════════════
def alpha_core(theta, S, f, N, a0):
    """The state-agnostic core of Eq 3 — identical across every state; the
    per-state modifier and α-band are what differentiate the regimes."""
    F_phi    = torch.exp(-torch.abs(theta - THETA_G) / SIGMA_TH)
    F_vortex = torch.exp(-((S - S_PHI) ** 2) / (2 * SIGMA_S ** 2))
    F_369 = torch.ones_like(theta)
    for h in HARM:
        F_369 = F_369 + KAPPA * torch.exp(-((f - h) ** 2) / (2 * SIGMA_F ** 2)) \
                              * torch.exp(-((N - h) ** 2) / (2 * SIGMA_N ** 2))
    return a0 * F_phi * F_vortex * F_369, F_phi, F_vortex, F_369

# ── State identifiers ────────────────────────────────────────────────────────
S_CLASSICAL   = 0
S_TOROIDAL    = 1
S_CUBIT       = 2
S_QUANTUM     = 3
S_QUBIT       = 4
S_PHANTOM     = 5
S_TEMPORAL    = 6
S_RELATIVISTIC= 7
S_HARMONIC    = 8
S_VACUUM      = 9
S_FRACTAL     = 10
S_RETROCAUSAL = 11
S_ELEMENTAL   = 12
S_HOLOGRAPHIC = 13
S_HOLOADS     = 14
S_THERMAL     = 15
S_MULTIVERSAL = 16
S_CONSCIOUSNESS = 17

# Width of per-state arrays (state_weights, mean_E_per_state, C_states rows).
# Must be strictly greater than the largest state id above (currently 17).
# Using a named constant instead of a bare literal 20 sprinkled through the
# code so that adding a state can't silently cause an out-of-bounds write —
# bump this in ONE place. (20 leaves headroom for two more states.)
N_STATE_SLOTS = 20
assert N_STATE_SLOTS > S_CONSCIOUSNESS, "N_STATE_SLOTS must exceed max state id"

STATE_NAMES = {
    S_CLASSICAL: 'CLASSICAL', S_TOROIDAL: 'TOROIDAL', S_CUBIT: 'CUBIT',
    S_QUANTUM: 'QUANTUM', S_QUBIT: 'QUBIT', S_PHANTOM: 'PHANTOM',
    S_TEMPORAL: 'TEMPORAL', S_RELATIVISTIC: 'RELATIVISTIC',
    S_HARMONIC: 'HARMONIC', S_VACUUM: 'VACUUM', S_FRACTAL: 'FRACTAL',
    S_RETROCAUSAL: 'RETROCAUSAL', S_ELEMENTAL: 'ELEMENTAL',
    S_HOLOGRAPHIC: 'HOLOGRAPHIC', S_HOLOADS: 'HOLOADS',
    S_THERMAL: 'THERMAL', S_MULTIVERSAL: 'MULTIVERSAL',
    S_CONSCIOUSNESS: 'CONSCIOUSNESS',
}

# α-range per state from the document's Preservation Factor Hierarchy
ALPHA_RANGES = {
    S_CLASSICAL:    (0.70, 0.90),
    S_TOROIDAL:     (0.92, 0.97),
    S_CUBIT:        (0.90, 0.96),
    S_QUANTUM:      (0.94, 0.98),
    S_QUBIT:        (0.97, 0.995),
    S_PHANTOM:      (1.00, 1.05),
    S_TEMPORAL:     (0.80, 0.93),
    S_RELATIVISTIC: (0.85, 0.94),
    S_HARMONIC:     (0.93, 0.98),
    S_VACUUM:       (0.96, 0.99),
    S_FRACTAL:      (0.95, 0.99),
    S_RETROCAUSAL:  (0.94, 0.98),
    S_ELEMENTAL:    (0.88, 0.95),
    S_HOLOGRAPHIC:  (0.98, 0.998),
    S_HOLOADS:      (0.98, 0.998),
    S_THERMAL:      (0.75, 0.92),
    S_MULTIVERSAL:  (0.94, 0.98),
    S_CONSCIOUSNESS:(0.90, 0.96),
}

# ═══════════════════════════════════════════════════════════════════════════════
# PER-STATE MODIFIERS — each returns a (n,) tensor that scales alpha_core
# into the state's specific physical regime.
# ═══════════════════════════════════════════════════════════════════════════════
def state_modifier(state_id, theta, S, f, N, E, a0, ctx):
    """
    Per-state modifier for Eq 3. Returns (n,) tensor ∈ (0, ~1.6].
    Each state has a UNIQUE physical computation that distinguishes it
    from Classical — not just a constant scale factor.
    """
    one = torch.ones_like(theta)

    if state_id == S_CLASSICAL:
        return one

    if state_id == S_THERMAL:
        # Boltzmann factor: exp(-E/kT). High T → modifier→1 (exploration).
        # Low T → modifier drops (exploitation). T from ctx.
        T = ctx.get('T', None)
        if isinstance(T, torch.Tensor):
            return torch.exp(-E / (T.clamp(min=0.1) * 3.0)).clamp(0.5, 1.2)
        return 0.9 * one

    if state_id == S_RELATIVISTIC:
        # Time dilation: 1/γ = √(1-v²/c²). High velocity → lower modifier
        # (slower subjective time = higher effective preservation).
        v2c2 = ctx.get('v2c2', None)
        if isinstance(v2c2, torch.Tensor):
            return torch.sqrt(1.0 - v2c2.clamp(0, 0.95)).clamp(0.7, 1.1)
        return 0.95 * one

    if state_id == S_TOROIDAL:
        # Closed-loop flow: reward oscillators whose state aligns with
        # the collective flow direction (toroidal major axis).
        flow_align = ctx.get('flow_align', None)
        if isinstance(flow_align, torch.Tensor):
            # Higher alignment with flow → higher modifier
            return (0.9 + 0.2 * flow_align.clamp(0, 1)).clamp(0.9, 1.1)
        return 1.0 * one

    if state_id == S_HARMONIC:
        # Resonance quality factor Q. Oscillators near 3-6-9 harmonics
        # get high Q → high modifier. This is DISTINCT from Classical
        # because it specifically rewards harmonic frequency matching.
        nearest = torch.stack([(f - h).abs() for h in HARM], 0).min(0).values
        Q = 1.0 / (0.15 + nearest)
        return (0.85 + 0.25 * Q / (Q + 1.0)).clamp(0.85, 1.15)

    if state_id == S_MULTIVERSAL:
        # Parallel-branch reinforcement. Best branch alignment → modifier.
        branch_gain = ctx.get('branch_gain', None)
        if isinstance(branch_gain, torch.Tensor):
            return branch_gain.clamp(0.9, 1.1)
        return torch.full_like(theta, ctx.get('branch_gain', 1.02)).clamp(0.9, 1.1)

    if state_id == S_RETROCAUSAL:
        # Reward oscillators whose past prediction was CORRECT about current state.
        pred_accuracy = ctx.get('pred_accuracy', None)
        if isinstance(pred_accuracy, torch.Tensor):
            # pred_accuracy ∈ [0,1]: 1 = perfect prediction of current state
            return (0.9 + 0.15 * pred_accuracy).clamp(0.9, 1.05)
        pc = ctx.get('pred_consistency', 0.5)
        return torch.full_like(theta, 0.9 + 0.1 * pc).clamp(0.9, 1.05)

    if state_id == S_FRACTAL:
        # Multi-scale coherence: reward oscillators whose state is consistent
        # across multiple timescales (fractal self-similarity).
        scale_coherence = ctx.get('scale_coherence', None)
        if isinstance(scale_coherence, torch.Tensor):
            return (0.9 + 0.15 * scale_coherence).clamp(0.9, 1.1)
        return 1.03 * one

    if state_id == S_VACUUM:
        # Zero-point stability: reward LOW-activity oscillators (reservoirs).
        # Vacuum state is for stable, quiet oscillators that preserve energy.
        vacuum_stability = ctx.get('vacuum_stability', None)
        if isinstance(vacuum_stability, torch.Tensor):
            return (0.9 + 0.15 * vacuum_stability).clamp(0.9, 1.1)
        return 1.02 * one

    if state_id == S_HOLOGRAPHIC:
        # Information preservation via boundary encoding.
        # Reward oscillators with high projection onto holographic boundary.
        holo_fidelity = ctx.get('holo_fidelity', None)
        if isinstance(holo_fidelity, torch.Tensor):
            return (0.95 + 0.1 * holo_fidelity).clamp(0.95, 1.1)
        sent = ctx.get('S_ent', 1.0)
        return torch.exp(torch.tensor(0.02 * sent, device=theta.device)) * one

    if state_id == S_HOLOADS:
        # Same as holographic but for AdS boundary compression
        holo_fidelity = ctx.get('holo_fidelity', None)
        if isinstance(holo_fidelity, torch.Tensor):
            return (0.95 + 0.1 * holo_fidelity).clamp(0.95, 1.1)
        return 1.02 * one

    if state_id == S_PHANTOM:
        # exp(w * ln(a)) where w < -1 → α > 1 (energy GAIN).
        # local_w is now properly dynamic: more negative when divergence is high.
        w = ctx.get('w', None)
        if isinstance(w, torch.Tensor):
            # w is in [-2.0, -1.0] range. For a_core ~ 0.85-0.95:
            # 0.9^(-1.5) = 1.17, 0.9^(-2.0) = 1.23
            # We compute the modifier as a_core^(-(|w|-1)) which gives > 1
            gain_exponent = (torch.abs(w) - 1.0).clamp(min=0.0, max=1.0)
            # Use a0 as base (typically 0.85-0.95)
            mod = torch.pow(1.0 / a0.clamp(min=0.8), gain_exponent)
            return mod.clamp(1.0, 1.15)
        return 1.05 * one

    if state_id == S_QUANTUM:
        # Quantum coherence: reward oscillators with high phase coherence
        # relative to their neighbors (entanglement proxy).
        phase_coherence = ctx.get('phase_coherence', None)
        if isinstance(phase_coherence, torch.Tensor):
            kappa = 0.15
            return (1.0 + kappa * phase_coherence).clamp(0.9, 1.15)
        # Fallback: frequency-based coherence
        kappa = 0.1
        Phi_sum = torch.zeros_like(one)
        for h in (3.0, 6.0, 9.0):
            Phi_h = torch.exp(-((f - h) ** 2) / 1.0) * torch.tanh(N / 5.0)
            Phi_sum = Phi_sum + Phi_h
        return (1.0 + kappa * Phi_sum).clamp(0.9, 1.15)

    if state_id == S_QUBIT:
        # Discrete quantum: Lorentzian peaks at harmonics (level transitions)
        kappa = 0.15
        P_sum = torch.zeros_like(one)
        gamma_w = 0.5
        for h in (3.0, 6.0, 9.0):
            P_h = (gamma_w ** 2) / ((f - h) ** 2 + gamma_w ** 2)
            P_h = P_h * torch.tanh(N / 5.0)
            P_sum = P_sum + P_h
        return (1.0 + kappa * P_sum).clamp(0.9, 1.15)

    if state_id == S_CONSCIOUSNESS:
        # Observer-coupled: attention-gated measurement
        A = ctx.get('A', 0.5)
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, device=theta.device)
        E_max = E.max().clamp(min=1e-6)
        I_ratio = E / E_max
        observer_factor = torch.exp(-I_ratio * 0.5)
        psi_collapse = 4.0 * I_ratio * (1.0 - I_ratio)
        kappa = 0.1
        return (observer_factor * (1.0 + kappa * psi_collapse * A)).clamp(0.8, 1.1)

    if state_id == S_TEMPORAL:
        # Relaxation envelope: monotonic in proper_time
        t_norm = ctx.get('t_norm', None)
        if isinstance(t_norm, torch.Tensor):
            return (0.85 + 0.2 * (1.0 - torch.exp(-t_norm / 0.3))).clamp(0.85, 1.1)
        t_val = ctx.get('t_norm', 0.5)
        return (0.85 + 0.2 * (1.0 - math.exp(-t_val / 0.3))) * one

    if state_id == S_CUBIT:
        # Discrete-space: lattice alignment bonus
        lattice_align = ctx.get('lattice_align', None)
        if isinstance(lattice_align, torch.Tensor):
            return (0.9 + 0.15 * lattice_align).clamp(0.9, 1.1)
        return 1.0 * one

    if state_id == S_ELEMENTAL:
        # Coupled-force gating: reward when multiple force channels align
        force_coupling = ctx.get('force_coupling', None)
        if isinstance(force_coupling, torch.Tensor):
            return (0.85 + 0.2 * force_coupling).clamp(0.85, 1.1)
        return 0.97 * one

    return one


def state_dissipation_modifier(state_id, theta, S, f, N, E, a0,
                               local_T=None, gamma=None, local_w=None,
                               vacuum_energy=None, scale_mem_coh=None,
                               proper_time_norm=None, holo_fidelity=None,
                               phase_coh=None, pred_acc=None, branch_gain=None,
                               force_coupling=None, lattice_align=None,
                               is_locked=None):
    """
    Per-state Eq 5 (Master Continuous) dissipation modifier, vectorized.

    Base (Classical) dissipation rate is Λ₀/(F_φ·F_vortex·F_369). This returns
    the per-oscillator multiplier M(state) on that base rate, one branch per
    regime, taken from each state's framework Eq 5 loss term. M<1 preserves
    better (slower decay); M>1 dissipates faster. Returns (n,) in (0.01, 3.0].
    """
    n = state_id.shape[0]
    dev = state_id.device
    M = torch.ones(n, device=dev)

    # Thermal — Arrhenius/Boltzmann loss exp(E_a/(k_B·T)): dissipation is HUGE
    # when cold (T→0 ⇒ exp→∞) and falls toward the base rate when hot. High T
    # is the exploration/barrier-crossing regime (low loss, search); cooling
    # forces settling (high loss). Activation energy E_a normalized to ~1.
    if local_T is not None:
        T = local_T.clamp(min=0.05, max=3.0)
        m_thermal = torch.exp(1.0 / T - 1.0).clamp(0.6, 3.0)
        M = torch.where(state_id == S_THERMAL, m_thermal, M)

    # Harmonic — Q replaces F_vortex: Λ₀/(F_φ·Q·F_369). Near 3-6-9 (measured as
    # f/F0, consistent with alpha_core) Q is high → low loss (resonant ringing).
    nearest = torch.stack([(f - h).abs() for h in HARM], 0).min(0).values
    Q = 1.0 / (0.15 + nearest)
    m_harm = (1.0 / Q.clamp(min=1e-3)).clamp(0.1, 2.0)
    M = torch.where(state_id == S_HARMONIC, m_harm, M)

    # Vacuum — 1/(1+Λ·R²) suppression; vacuum_energy stands in for the Λ·R²
    # reservoir term. Larger reservoir → smaller loss.
    if vacuum_energy is not None:
        m_vac = (1.0 / (1.0 + 2.0 * vacuum_energy.clamp(min=0.0))).clamp(0.3, 1.0)
        M = torch.where(state_id == S_VACUUM, m_vac, M)

    # Toroidal — R_minor/R_major closed-loop ratio (<1). Approximated by the
    # canonical torus aspect ratio; no per-oscillator poloidal/toroidal split.
    M = torch.where(state_id == S_TOROIDAL, torch.full((n,), 0.7, device=dev), M)

    # Relativistic — 1/γ time dilation (√(-g) reduced to scalar γ). Fast-changing
    # oscillators have high γ, so their loss per network tick is suppressed.
    if gamma is not None:
        m_rel = (1.0 / gamma.clamp(min=1.0)).clamp(0.3, 1.0)
        M = torch.where(state_id == S_RELATIVISTIC, m_rel, M)

    # Holographic / HoloAdS — exp(−S_ent/k_B): entanglement entropy drives the
    # loss to near zero. This is the long-term-memory mechanism. Boundary
    # fidelity is the entanglement proxy (high fidelity ⇒ high S_ent ⇒ ~0 loss).
    if holo_fidelity is not None:
        S_ent = 2.0 + 3.0 * holo_fidelity.clamp(0, 1)
        m_holo = torch.exp(-S_ent).clamp(0.01, 0.5)
    else:
        m_holo = torch.full((n,), 0.05, device=dev)
    M = torch.where(state_id == S_HOLOGRAPHIC, m_holo, M)
    M = torch.where(state_id == S_HOLOADS, m_holo, M)

    # Phantom — |1+w| with w<-1: the loss term shrinks to near zero (the only
    # gain regime). Energy gain itself is applied through α∈[1.00,1.05] in Eq 4.
    if local_w is not None:
        m_phan = torch.abs(1.0 + local_w).clamp(0.0, 0.5)
    else:
        m_phan = torch.full((n,), 0.1, device=dev)
    M = torch.where(state_id == S_PHANTOM, m_phan, M)

    # Fractal — λ^{D_f} with golden scaling λ=1/φ and D_f∈[1,2] from cross-scale
    # coherence. Multi-scale redundancy makes loss sub-Classical.
    if scale_mem_coh is not None:
        D_f = 1.0 + scale_mem_coh.clamp(0, 1)
        m_frac = ((1.0 / PHI) ** D_f).clamp(0.3, 0.9)
    else:
        m_frac = torch.full((n,), 0.6, device=dev)
    M = torch.where(state_id == S_FRACTAL, m_frac, M)

    # Quantum / Qubit — Lindblad decoherence: the dissipator γ_b0 is suppressed
    # by coherence. High phase coherence ⇒ low loss (a coherent superposition
    # resists information loss). Coherence-driven, not a flat constant. Qubit
    # gets an extra discreteness factor (discrete levels prune leakage paths).
    if phase_coh is not None:
        coh = phase_coh.clamp(0, 1)
        m_quant = (1.0 - 0.6 * coh).clamp(0.3, 1.0)
        M = torch.where(state_id == S_QUANTUM, m_quant, M)
        M = torch.where(state_id == S_QUBIT, (m_quant * 0.9).clamp(0.25, 1.0), M)
    else:
        M = torch.where(state_id == S_QUANTUM, torch.full((n,), 0.6, device=dev), M)
        M = torch.where(state_id == S_QUBIT, torch.full((n,), 0.5, device=dev), M)

    # Temporal — (1−exp(−t/τ_relax)) relaxes loss downward with time-in-state:
    # recently-routed oscillators dissipate near base rate; persistent ones
    # earn lower loss (maturation).
    if proper_time_norm is not None:
        m_temp = (1.0 - 0.5 * (1.0 - torch.exp(-proper_time_norm.clamp(min=0)))).clamp(0.5, 1.0)
        M = torch.where(state_id == S_TEMPORAL, m_temp, M)

    # Cubit — discrete space prunes loss paths: Λ₀·(1−ρ_discrete)/(…), where the
    # lattice alignment is how well the state sits on a discrete cell. Strong
    # lattice alignment ⇒ fewer leakage channels ⇒ lower loss than Classical.
    if lattice_align is not None:
        m_cubit = (1.0 - 0.35 * lattice_align.clamp(0, 1)).clamp(0.6, 1.0)
        M = torch.where(state_id == S_CUBIT, m_cubit, M)
    else:
        M = torch.where(state_id == S_CUBIT, torch.full((n,), 0.85, device=dev), M)

    # Elemental — Σ_i Λ_i·α_i(Q²): summed loss over the four force channels. When
    # the channels are well-coupled (all aligned) the effective loss drops; the
    # geometric-mean coupling stands in for the running-coupling product.
    if force_coupling is not None:
        m_elem = (1.0 - 0.3 * force_coupling.clamp(0, 1)).clamp(0.6, 1.1)
        M = torch.where(state_id == S_ELEMENTAL, m_elem, M)
    else:
        M = torch.where(state_id == S_ELEMENTAL, torch.full((n,), 0.95, device=dev), M)

    # Multiversal — cross-branch transfer Σ_j T_ij²·(…): a branch coupled to many
    # good parallel branches loses less (redundancy across worlds). branch_gain
    # is the best-branch alignment proxy for the tunneling coupling.
    if branch_gain is not None:
        m_multi = (1.0 - 0.25 * (branch_gain.clamp(0.9, 1.1) - 0.9) / 0.2).clamp(0.7, 1.0)
        M = torch.where(state_id == S_MULTIVERSAL, m_multi, M)
    else:
        M = torch.where(state_id == S_MULTIVERSAL, torch.full((n,), 0.9, device=dev), M)

    # Retrocausal — future-kernel ∫K_retro·E(t')dt', approximated causally by
    # prediction accuracy: an oscillator whose prediction proved correct is on a
    # stable predicted trajectory and dissipates less.
    if pred_acc is not None:
        m_retro = (1.0 - 0.3 * pred_acc.clamp(0, 1)).clamp(0.6, 1.0)
        M = torch.where(state_id == S_RETROCAUSAL, m_retro, M)
    else:
        M = torch.where(state_id == S_RETROCAUSAL, torch.full((n,), 0.9, device=dev), M)

    # Consciousness — exp(I_obs/I_max) attention gate: attended oscillators (high
    # observer information) dissipate less. A(t) here is a global proxy, so this
    # is a coarse salience gate rather than per-oscillator attention.
    M = torch.where(state_id == S_CONSCIOUSNESS, torch.full((n,), 0.9, device=dev), M)

    # Locked reference nodes are consolidated boundary memory — force the
    # holographic near-zero loss regardless of routed state.
    if is_locked is not None:
        M = torch.where(is_locked, torch.full((n,), 0.02, device=dev), M)

    return M.clamp(0.01, 3.0)


def alpha_for_state(state_id, theta, S, f, N, E, a0, ctx, core_pack=None):
    """
    Full Eq 3 for a given state.
    Core × modifier, mapped into the state's hierarchy band.

    core_pack: optional (core, F_phi, F_vortex, F_369) tuple from a prior
    alpha_core() call. The core is state-INDEPENDENT (only the modifier and
    α-band differ per state), so loops that evaluate many states per tick
    must compute it once and pass it in — recomputing it per state was ~34
    identical alpha_core() evaluations per dynamics tick.
    """
    if core_pack is not None:
        core, F_phi, F_vortex, F_369 = core_pack
    else:
        core, F_phi, F_vortex, F_369 = alpha_core(theta, S, f, N, a0)
    mod = state_modifier(state_id, theta, S, f, N, E, a0, ctx)
    raw = core * mod
    # Map into the state's α-range
    lo, hi = ALPHA_RANGES.get(state_id, (0.7, 0.9))
    # raw is typically in [0, ~1.2]. Normalize to [0,1] then scale to band.
    quality = raw.clamp(0, 1.5) / 1.5   # normalized quality ∈ [0,1]
    alpha = lo + (hi - lo) * quality
    return alpha, F_phi, F_vortex, F_369


# ═══════════════════════════════════════════════════════════════════════════════
# Eq 6 — VALENCE (the objective the field maximizes)
# ═══════════════════════════════════════════════════════════════════════════════
def valence(s, R, I, alpha, lam=0.25):
    """V(s) = α⟨s,R⟩ − λ‖s−I‖²"""
    align = (s * R).sum(-1)
    dev = ((s - I) ** 2).sum(-1)
    return alpha * align - lam * dev


# ═══════════════════════════════════════════════════════════════════════════════
# SENSORY PROJECTION — fixed random projection from raw input to d-dim
# ═══════════════════════════════════════════════════════════════════════════════
class SensoryProjection:
    """Fixed random projection from raw input space to n_sensory dimensions."""
    def __init__(self, raw_dim, n_sensory, seed=42):
        rng = torch.Generator().manual_seed(seed)
        self.W = torch.randn(raw_dim, n_sensory, generator=rng) / math.sqrt(raw_dim)
        self.n_sensory = n_sensory

    def __call__(self, x):
        """Project raw input x (raw_dim,) → (n_sensory,) normalized."""
        x = x.to(self.W.device)
        out = x @ self.W
        return out / out.norm().clamp(min=1e-6)

    def to(self, device):
        """Move projection matrix to specified device."""
        self.W = self.W.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════
def spread_directions(n, d, device, seed=0):
    """Generate n well-spread unit vectors in d dimensions."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    v = torch.randn(n, d, generator=g, device='cpu').to(device)
    # Gram-Schmidt-ish orthogonalization for small n
    if n <= d:
        Q, _ = torch.linalg.qr(v.T)
        return Q.T[:n]
    return v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — FIELD SUBSTRATE
# ═══════════════════════════════════════════════════════════════════════════════
class UERFField:
    """
    The brain: a field of n_max coupled oscillators, each with d-dim state.
    Implements all 6 equations from the UERF framework.
    """
    # States that participate in routing competition
    CANDIDATE_STATES = [
        S_CLASSICAL, S_TOROIDAL, S_CUBIT, S_QUANTUM, S_QUBIT,
        S_PHANTOM, S_TEMPORAL, S_RELATIVISTIC, S_HARMONIC,
        S_VACUUM, S_FRACTAL, S_RETROCAUSAL, S_ELEMENTAL,
        S_HOLOGRAPHIC, S_THERMAL, S_MULTIVERSAL,
        S_CONSCIOUSNESS,
    ]
    # S_HOLOADS (id 14) is excluded from routing — it is not one of the 17 doc
    # states (14 canonical + 3 speculative). Its code branches remain intact so
    # any checkpointed oscillators in that state still compute correct physics;
    # subsequent routing ticks will migrate them to Holographic naturally.

    def __init__(self, n_max=200, n_initial=100, d=32, input_dim=None,
                 n_classes=None, device=None):
        dev = device or DEVICE
        self.device = dev
        self.n_max = n_max
        n_initial = min(n_initial, n_max)  # Safety: can't init more than capacity
        self.d = d
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.experience_count = 0
        self.births = 0
        self.deaths = 0
        self.crystallization_events = 0

        # ── Identity parameters (per oscillator)
        self.theta = torch.randn(n_max, device=dev) * 0.8 + THETA_G
        self.S     = torch.randn(n_max, device=dev) * 0.5 + S_PHI
        self.f     = torch.rand(n_max, device=dev) * 8.0 + 1.0
        self.N     = torch.rand(n_max, device=dev) * 8.0 + 1.0
        self.a0    = torch.rand(n_max, device=dev) * 0.15 + 0.85
        self.age   = torch.zeros(n_max, device=dev)
        self.alive_mask = torch.zeros(n_max, dtype=torch.bool, device=dev)
        self.alive_mask[:n_initial] = True

        # ── d-dim state vectors
        self.s = torch.zeros(n_max, d, device=dev)
        cr = torch.randn(n_max, d, device=dev)
        self.c = cr / cr.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # ── Phase (Kuramoto)
        ph = torch.rand(n_max, device=dev) * 2 * math.pi
        self.phase_vec = torch.stack([torch.cos(ph), torch.sin(ph)], 1)

        # ── Emergent state assignment
        self.state_id = torch.full((n_max,), S_CLASSICAL, dtype=torch.long, device=dev)

        # ── Bond tensor C[i,j] is (d,d) matrix for each pair
        self.C = torch.zeros(n_max, n_max, d, d, device=dev)
        self.C_mask = torch.zeros(n_max, n_max, dtype=torch.bool, device=dev)

        # ── FIX 11: Self-interactions (autapses) per Paper [3]
        # Initialize diagonal bonds with small identity-like matrices
        # This reshapes the energy landscape to confine dynamics to stored patterns
        for i in range(n_initial):
            self.C[i, i] = torch.eye(d, device=dev) * 0.05

        # ── Slot tags
        self._is_sensory  = torch.zeros(n_max, dtype=torch.bool, device=dev)
        self._is_teaching = torch.zeros(n_max, dtype=torch.bool, device=dev)
        self.t_start = self.t_end = None

        if input_dim is not None:
            self.theta[:input_dim] = THETA_G
            self.S[:input_dim]     = S_PHI
            self.f[:input_dim]     = 6.0
            self.N[:input_dim]     = torch.arange(input_dim, device=dev).float() % 9 + 1
            self.a0[:input_dim]    = 0.99
            self.age[:input_dim]   = 1e9
            self.alive_mask[:input_dim] = True
            self._is_sensory[:input_dim] = True
            self.c[:input_dim] = spread_directions(input_dim, d, dev, 0)

        if n_classes is not None and input_dim is not None:
            t0, t1 = input_dim, input_dim + n_classes
            self.theta[t0:t1] = THETA_G
            self.S[t0:t1]     = S_PHI
            self.f[t0:t1]     = 6.0
            self.N[t0:t1]     = torch.arange(n_classes, device=dev).float() % 9 + 1
            self.a0[t0:t1]    = 0.99
            self.age[t0:t1]   = 1e9
            self.alive_mask[t0:t1] = True
            self._is_teaching[t0:t1] = True
            self.c[t0:t1] = spread_directions(n_classes, d, dev, 1000)
            self.t_start, self.t_end = t0, t1

            # ── HOLOGRAPHIC BOUNDARY PROJECTOR
            with torch.no_grad():
                d_boundary = max(2, min(n_classes, d // 3))
                _, _, Vt = torch.linalg.svd(self.c[t0:t1], full_matrices=False)
                B = Vt[:d_boundary].T
                self._holo_projector = (B @ B.T).to(dev)
                self._holo_area_limit = float(d_boundary) / float(d)
        else:
            self._holo_projector = None
            self._holo_area_limit = 1.0

        # ── State-specific auxiliary variables
        self.proper_time = torch.zeros(n_max, device=dev)
        self.gamma = torch.ones(n_max, device=dev)
        self.local_T = torch.ones(n_max, device=dev) * 0.5
        self.local_v2c2 = torch.zeros(n_max, device=dev)
        self.local_w = torch.full((n_max,), -1.0, device=dev)  # Neutral: PHANTOM only for genuinely divergent
        self.vacuum_energy = torch.ones(n_max, device=dev) * 0.1
        self.force_channels = torch.zeros(n_max, 4, device=dev)
        self.future_prediction = torch.zeros(n_max, d, device=dev)
        self._prev_prediction = torch.zeros(n_max, d, device=dev)
        self.scale_memory = torch.zeros(n_max, 4, d, device=dev)
        self.s_branches = torch.randn(n_max, 4, d, device=dev) * 0.01
        self.branch_weights = torch.ones(n_max, 4, device=dev) * 0.25
        self._E_history = torch.zeros(n_max, 8, device=dev)
        self._prev_s = None
        self._prev_mag = None

        # ── Cross-state coupling matrix (for Unified Master Equation)
        # Pairs from the doc's Cross-State Coupling section + one defensible extra.
        # No integer-adjacency coupling — state IDs are arbitrary, so i±1 has no
        # physical meaning.
        n_states = N_STATE_SLOTS
        self.C_states = torch.zeros(n_states, n_states, device=dev)
        _COUPLINGS = [
            # Doc-specified hybrids
            (S_QUANTUM,     S_RELATIVISTIC, 0.05),  # Dirac eq in curved spacetime
            (S_THERMAL,     S_FRACTAL,      0.03),  # heat diffusion on fractal geometry
            (S_HARMONIC,    S_TOROIDAL,     0.05),  # tokamak plasma resonances
            (S_HOLOGRAPHIC, S_ELEMENTAL,    0.03),  # AdS/CFT + gauge theory coupling
            (S_PHANTOM,     S_VACUUM,       0.05),  # dark energy from quantum fluctuations
            (S_TEMPORAL,    S_THERMAL,      0.03),  # non-equilibrium thermodynamics
            (S_FRACTAL,     S_HARMONIC,     0.03),  # multi-scale resonance structures
            # Physically defensible extra (not in doc)
            (S_PHANTOM,     S_HOLOGRAPHIC,  0.05),  # phantom growth → holographic consolidation
        ]
        for _a, _b, _w in _COUPLINGS:
            self.C_states[_a, _b] = _w
            self.C_states[_b, _a] = _w

        self.state_weights = torch.ones(n_states, device=dev) / n_states

        # ── Replay buffer for continual learning — STRATIFIED PER CLASS
        # Old FIFO buffer caused catastrophic forgetting: by step 750 of
        # Phase B, buffer was 100% Phase B samples, rehearsal reinforced
        # only Phase B. Now: per-class deque, keep N samples per class,
        # rehearsal samples uniformly across all classes seen.
        self._replay_per_class = {}            # dict[int, list of (x, tv)]
        self._replay_capacity_per_class = 20
        # DISABLE FLAG: when True, replay buffer doesn't accumulate samples
        # and rehearsal is skipped. Required for clean physics-only continual
        # learning test — without it, the patent claim is "physics + replay"
        # not "physics alone."
        self._replay_disabled = False

        # ── ABLATION TOGGLES ──────────────────────────────────────────────
        # Each of the 6 audit fixes is independently switchable so we can run
        # the full 64-combination ablation from a single brain file (no copy-
        # paste drift between variants). Default all-False = original behavior.
        self.fixes = {
            'fix1': False, 'fix2': False, 'fix3': False,
            'fix4': False, 'fix5': False, 'fix6': False,
        }
        self._last_birth_step = -1000

        # ── Per-class teaching bond TRUE-MEAN accumulators
        # Stores Σ outer(c_teach, s_sensor) per class; final bond = sum/count.
        # Phase B class learning NEVER touches Phase A class accumulators
        # because the active-class check is per-experience.
        self._teach_bond_sum = (torch.zeros(n_classes, n_max, d, d, device=dev)
                                if n_classes else None)
        self._teach_bond_count = torch.zeros(n_classes if n_classes else 10, device=dev)

        # ── Class consolidation locks (DeepSeek #3 + framework crystallization)
        # _class_locked[i] = True → osc i frozen as part of a consolidated
        # class representation. Locked osc: no decay, no competitive drift,
        # bonds protected from pruning, high budget.
        self._class_locked = torch.zeros(n_max, dtype=torch.bool, device=dev)
        # Map: which class each locked oscillator belongs to (−1 if not locked)
        self._locked_class = torch.full((n_max,), -1, dtype=torch.long, device=dev)

        # ── Novelty tracking
        self._input_running_mean = None
        self._input_running_var = torch.tensor(1.0, device=dev)

        # ── FIX 3: Crystallization requires sustained high valence
        self._valence_accumulator = torch.zeros(n_max, device=dev)
        self._valence_count = torch.zeros(n_max, device=dev)
        # Threshold on the EMA valence (not EMA/count — see crystallized()).
        # Interior magnitudes sit well below 1 in normal training, so the
        # sustained-valence band is ~0.1-0.5; 0.25 demands genuinely strong,
        # persistent alignment without being unreachable like the old 0.6
        # (which was compared against EMA/count ≈ 0).
        self._crystallization_threshold = 0.25

    # ── Helpers ──
    def energy(self):
        return (self.s ** 2).sum(-1)

    def state_magnitude(self):
        return self.s.norm(dim=-1)

    def phase_angle(self):
        return torch.atan2(self.phase_vec[:, 1], self.phase_vec[:, 0])

    def eq5_decay_factor(self, dt=0.08, exponent_cap=0.5, ctx=None,
                         precomputed=None):
        """
        Eq 5 (Master Continuous) dissipation as a per-tick decay factor,
        state-specific.

            dE/dt = P_in − (Λ₀·M_state / (F_φ·F_vortex·F_369))·E

        M_state is the per-regime modifier (state_dissipation_modifier). The
        homogeneous solution over one tick is E·exp(−rate·dt); since the field
        scales s (E=‖s‖²) the factor on s is the sqrt. exponent_cap bounds
        single-tick loss (P_in balances it in the full equation). To avoid
        recomputing F_φ·F_vortex·F_369 every call, the dynamics loop may pass
        `precomputed=(F_phi,F_vortex,F_369)` from its routing pass. Returns an
        (n,) factor in (0, 1].
        """
        if precomputed is not None:
            F_phi, F_vortex, F_369 = precomputed
        else:
            _, F_phi, F_vortex, F_369 = alpha_core(
                self.theta, self.S, self.f, self.N, self.a0)
        denom = (F_phi * F_vortex * F_369).clamp(min=1e-3)

        g = ctx if isinstance(ctx, dict) else {}

        def _t(key):
            v = g.get(key, None)
            return v if isinstance(v, torch.Tensor) else None

        M_state = state_dissipation_modifier(
            self.state_id, self.theta, self.S, self.f, self.N, self.energy(),
            self.a0,
            local_T=self.local_T, gamma=self.gamma, local_w=self.local_w,
            vacuum_energy=self.vacuum_energy,
            scale_mem_coh=_t('scale_coherence'),
            proper_time_norm=_t('t_norm'),
            holo_fidelity=_t('holo_fidelity'),
            phase_coh=_t('phase_coherence'),
            pred_acc=_t('pred_accuracy'),
            branch_gain=_t('branch_gain'),
            force_coupling=_t('force_coupling'),
            lattice_align=_t('lattice_align'),
            is_locked=self._class_locked,
        )

        exponent = (LAMBDA0 * M_state * dt / denom).clamp(max=exponent_cap)
        return torch.sqrt(torch.exp(-exponent))

    def crystallized(self):
        """
        FIX 3: Crystallization requires BOTH golden-ratio proximity AND
        sustained high valence (accumulated over time). This prevents
        premature crystallization from identity drift alone.
        """
        golden_proximity = (
            (torch.abs(self.theta - THETA_G) < 0.10) &
            (torch.abs(self.S - S_PHI) < 0.10)
        )
        # Must have accumulated sufficient positive valence. The accumulator
        # is an EMA (0.95/0.05 in _identity_drift) and already IS the running
        # mean — dividing it by the ever-growing update count drove the value
        # toward zero within a few steps and made the threshold permanently
        # unreachable (root cause of locked=0 / crystallized=0 in Phase 1).
        mean_valence = self._valence_accumulator
        valence_sufficient = mean_valence > self._crystallization_threshold
        return golden_proximity & valence_sufficient & self.alive_mask

    def report(self):
        alive = self.alive_mask
        interior = alive & ~self._is_sensory & ~self._is_teaching
        cr = self.crystallized()
        dist = {}
        for sid in self.CANDIDATE_STATES:
            c_ = int(((self.state_id == sid) & interior).sum())
            if c_ > 0:
                dist[STATE_NAMES.get(sid, f'S{sid}')] = c_
        return {
            'alive': int(alive.sum()),
            'interior': int(interior.sum()),
            'crystallized': int(cr.sum()),
            'crystallization_events': self.crystallization_events,
            'births': self.births,
            'deaths': self.deaths,
            'bonds': int(self.C_mask.sum()),
            'mean_E': float(self.energy()[alive].mean()) if alive.any() else 0.0,
            'global_order': float((self.s[alive].mean(0)).norm()) if alive.any() else 0.0,
            'experience_count': self.experience_count,
            'state_dist': dist,
            'd': self.d,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — DYNAMICS + EMERGENT STATE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
class UERFDynamics:
    """Mixin: dynamics methods for UERFField."""

    def _build_routing_ctx(self):
        """
        Build the context dict with per-oscillator physics tensors for routing.
        FIX 4: Each state gets genuinely distinct input signals so they can
        differentiate themselves in the routing competition.
        """
        s = self.s
        mag = self.state_magnitude()
        n = self.n_max
        d = self.d
        ctx = {}

        # Thermal: local temperature (energy variance)
        ctx['T'] = self.local_T

        # Relativistic: velocity squared
        ctx['v2c2'] = self.local_v2c2

        # Phantom: divergence-driven w
        ctx['w'] = self.local_w

        # Temporal: normalized proper time
        pt_max = self.proper_time.max().clamp(min=1.0)
        ctx['t_norm'] = self.proper_time / pt_max

        # Toroidal: flow alignment (how much each osc aligns with collective)
        interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
        if interior.any() and mag[interior].sum() > 0:
            flow_dir = s[interior].sum(0)
            flow_norm = flow_dir.norm().clamp(min=1e-8)
            flow_dir = flow_dir / flow_norm
            flow_align = (s * flow_dir.unsqueeze(0)).sum(-1) / mag.clamp(min=1e-8)
            ctx['flow_align'] = flow_align.clamp(-1, 1)
        else:
            ctx['flow_align'] = torch.zeros(n, device=self.device)

        # Harmonic: already uses f directly in modifier

        # Holographic: projection fidelity, bounded by the boundary area.
        # The holographic principle caps boundary-encoded information by the
        # boundary's fractional dimension (_holo_area_limit = d_boundary/d).
        # We report fidelity as the share of boundary capacity the node
        # saturates, so the Holographic reward enforces the boundary condition
        # (a node can't exceed what the boundary can hold) rather than treating
        # the regime as a mere flag.
        if self._holo_projector is not None:
            s_proj = s @ self._holo_projector
            proj_fidelity = (s * s_proj).sum(-1) / (mag ** 2).clamp(min=1e-8)
            area_limit = getattr(self, '_holo_area_limit', 1.0)
            if area_limit and area_limit > 0:
                # Fraction of the boundary's capacity that is occupied.
                proj_fidelity = (proj_fidelity / area_limit)
            ctx['holo_fidelity'] = proj_fidelity.clamp(0, 1)
        else:
            ctx['holo_fidelity'] = torch.zeros(n, device=self.device)

        # Retrocausal: FIX 2 — prediction ACCURACY (not self-correlation)
        # How well did our PREVIOUS prediction match CURRENT state?
        if self._prev_prediction is not None:
            pred_norm = self._prev_prediction.norm(dim=-1).clamp(min=1e-8)
            s_norm = mag.clamp(min=1e-8)
            # Cosine similarity between prediction and actual
            cos_sim = (self._prev_prediction * s).sum(-1) / (pred_norm * s_norm)
            # Subtract baseline (mean similarity) to remove bias
            baseline = cos_sim[self.alive_mask].mean() if self.alive_mask.any() else 0.0
            ctx['pred_accuracy'] = (cos_sim - baseline).clamp(0, 1)
        else:
            ctx['pred_accuracy'] = torch.zeros(n, device=self.device)

        # Fractal: multi-scale coherence
        if self.scale_memory.abs().sum() > 0:
            # How consistent is current state across timescales?
            coherences = []
            for k in range(1, 4):
                sm_k = self.scale_memory[:, k]
                sm_norm = sm_k.norm(dim=-1).clamp(min=1e-8)
                cos_k = (s * sm_k).sum(-1) / (mag.clamp(min=1e-8) * sm_norm)
                coherences.append(cos_k)
            scale_coh = torch.stack(coherences, dim=-1).mean(dim=-1)
            ctx['scale_coherence'] = scale_coh.clamp(0, 1)
        else:
            ctx['scale_coherence'] = torch.zeros(n, device=self.device)

        # Vacuum: stability (inverse of recent change)
        if self._prev_s is not None:
            change = (s - self._prev_s).norm(dim=-1)
            stability = torch.exp(-change * 5.0)  # high when stable
            ctx['vacuum_stability'] = stability
        else:
            ctx['vacuum_stability'] = torch.ones(n, device=self.device) * 0.5

        # Quantum: phase coherence with neighbors — bounded to the alive
        # high-water mark (dormant slots have no bonds, so the full n×n
        # phase-difference matrix wasted O(n_max²) work every routing pass).
        phi = self.phase_angle()
        if self.C_mask.any():
            alive_idx = self.alive_mask.nonzero(as_tuple=True)[0]
            n_hi = int(alive_idx.max().item()) + 1 if len(alive_idx) > 0 else 1
            phi_hi = phi[:n_hi]
            # Mean phase difference with bonded neighbors
            phase_diff = torch.cos(phi_hi.unsqueeze(0) - phi_hi.unsqueeze(1))
            mask_hi = self.C_mask[:n_hi, :n_hi].float()
            bonded_coherence = (phase_diff * mask_hi).sum(1)
            n_bonds = mask_hi.sum(1).clamp(min=1)
            coh = torch.zeros(n, device=self.device)
            coh[:n_hi] = (bonded_coherence / n_bonds).clamp(0, 1)
            ctx['phase_coherence'] = coh
        else:
            ctx['phase_coherence'] = torch.zeros(n, device=self.device)

        # Multiversal: best branch alignment
        if self.s_branches.abs().sum() > 0:
            # Which branch best predicts current state?
            branch_aligns = torch.zeros(n, 4, device=self.device)
            for j in range(4):
                sb = self.s_branches[:, j]
                sb_norm = sb.norm(dim=-1).clamp(min=1e-8)
                branch_aligns[:, j] = (sb * s).sum(-1) / (sb_norm * mag.clamp(min=1e-8))
            best_branch = branch_aligns.max(dim=-1).values
            ctx['branch_gain'] = (0.95 + 0.1 * best_branch).clamp(0.9, 1.1)
        else:
            ctx['branch_gain'] = torch.ones(n, device=self.device)

        # Cubit: lattice alignment
        k_lattice = 4
        n_c = torch.floor(mag * k_lattice).clamp(0, k_lattice - 1).long()
        idx = torch.arange(d, device=self.device).unsqueeze(0)
        roll_idx = (idx - n_c.unsqueeze(-1)) % d
        R_cubit = torch.gather(self.c, 1, roll_idx)
        lattice_align = (s * R_cubit).sum(-1) / mag.clamp(min=1e-8)
        ctx['lattice_align'] = lattice_align.clamp(0, 1)

        # Elemental: force channel coupling
        if self.force_channels.abs().sum() > 0:
            # Product of normalized force couplings (high when all forces align)
            fc_norm = self.force_channels / self.force_channels.max(dim=0).values.clamp(min=1e-8)
            coupling = fc_norm.prod(dim=-1) ** 0.25  # geometric mean
            ctx['force_coupling'] = coupling.clamp(0, 1)
        else:
            ctx['force_coupling'] = torch.zeros(n, device=self.device)

        # Consciousness: attention proxy
        active_frac = (mag > 0.1).float().mean()
        ctx['A'] = active_frac

        return ctx

    def _route_states(self, ctx):
        """
        EMERGENT ROUTING by VALENCE (Eq 6) with STOCHASTIC ASSIGNMENT.
        
        The key insight: when oscillator magnitudes are small (early training),
        valence differences between states are tiny. Pure argmax collapses
        everything to one state. Instead, we use STOCHASTIC routing:
        - Each oscillator samples its state from the softmax distribution
        - This ensures state DIVERSITY while still respecting valence ordering
        - As training progresses and magnitudes grow, the distribution sharpens
          naturally (larger magnitudes → larger valence differences → more
          deterministic routing) — the physics self-anneals.
        
        Additionally, each state computes a DISTINCT fitness signal that uses
        per-oscillator properties (frequency, phase, connectivity) so that
        different oscillators genuinely prefer different states.
        """
        interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
        # CRITICAL: locked oscillators MUST keep their assigned HOLOGRAPHIC state.
        # If they're routed normally, the very next experience() call will
        # reassign them based on whatever signal is loudest, erasing the
        # consolidation. Exclude them from routing entirely — they stay
        # frozen in whatever state consolidate_class put them in.
        routable = interior & ~self._class_locked
        n = self.n_max
        d = self.d
        s = self.s
        c = self.c
        mag = self.state_magnitude()
        lam = 0.25

        if not interior.any():
            return s, torch.zeros(n, device=self.device)

        n_int = int(interior.sum())

        # Build full context if not provided
        if ctx is None or len(ctx) == 0:
            ctx = self._build_routing_ctx()

        # Compute FITNESS for each candidate state.
        # Fitness = α_state * state_specific_signal - lam * deviation
        # Each state uses a UNIQUE signal derived from oscillator properties.
        n_states = len(self.CANDIDATE_STATES)
        fitness = torch.zeros(n, n_states, device=self.device)

        # Base alignment (shared)
        base_align = (s * c).sum(-1)  # ⟨s, c⟩
        I_k = c * mag.unsqueeze(-1)
        base_dev = ((s - I_k) ** 2).sum(-1)

        # alpha_core is state-independent — compute ONCE for all 17 states
        # (and reuse below for best_a). Energy too.
        _core_pack = alpha_core(self.theta, self.S, self.f, self.N, self.a0)
        _E_now = self.energy()

        for k, sid in enumerate(self.CANDIDATE_STATES):
            alpha_k, _, _, _ = alpha_for_state(
                sid, self.theta, self.S, self.f, self.N,
                _E_now, self.a0, ctx, core_pack=_core_pack
            )

            # STATE-SPECIFIC FITNESS SIGNALS
            # Each state has a unique criterion that makes it prefer
            # certain oscillators over others.
            if sid == S_CLASSICAL:
                # Classical: baseline, no bonus
                signal = base_align
                dev = base_dev

            elif sid == S_HOLOGRAPHIC and self._holo_projector is not None:
                # Holographic: boundary projection fidelity
                s_proj = s @ self._holo_projector
                signal = (s * s_proj).sum(-1)
                dev = ((s - s_proj) ** 2).sum(-1)

            elif sid == S_TOROIDAL:
                # Toroidal: flow alignment (collective direction)
                flow_align = ctx.get('flow_align', torch.zeros(n, device=self.device))
                signal = base_align + 0.5 * flow_align * mag
                dev = base_dev

            elif sid == S_HARMONIC:
                # Harmonic: frequency proximity to 3-6-9
                freq_bonus = torch.zeros(n, device=self.device)
                for h in HARM:
                    freq_bonus += torch.exp(-((self.f - h) ** 2) / 1.5)
                signal = base_align * (1.0 + 0.5 * freq_bonus)
                dev = base_dev

            elif sid == S_FRACTAL:
                # Fractal: multi-scale coherence
                scale_coh = ctx.get('scale_coherence', torch.zeros(n, device=self.device))
                signal = base_align * (1.0 + 0.5 * scale_coh)
                dev = base_dev

            elif sid == S_RETROCAUSAL:
                # Retrocausal: prediction accuracy
                pred_acc = ctx.get('pred_accuracy', torch.zeros(n, device=self.device))
                signal = base_align + 0.5 * pred_acc * mag
                dev = base_dev

            elif sid == S_VACUUM:
                # Vacuum: zero-point reservoir for stable resonators.
                # CRITICAL: must require GENUINE alignment, not just stillness.
                # Without this floor, decay_factor (0.85) creates "quiet" osc
                # that have small change → high stability → routes to VACUUM,
                # which has highest α (0.99) → preserves more energy → stays
                # quiet → routing collapses to all-VACUUM. The base_align
                # floor breaks the runaway.
                stability = ctx.get('vacuum_stability', torch.ones(n, device=self.device) * 0.5)
                # Only reward stability if the osc is genuinely active
                # (base_align > floor). Otherwise treat as classical.
                vacuum_reward = torch.where(
                    base_align > 0.1,
                    base_align * (1.0 + 0.3 * stability),   # genuine vacuum: reward
                    base_align * 0.5,                        # quiescence: penalize
                )
                signal = vacuum_reward
                dev = base_dev

            elif sid == S_THERMAL:
                # Thermal: high local temperature (high energy variance)
                T = ctx.get('T', torch.ones(n, device=self.device) * 0.5)
                T_norm = T / T.max().clamp(min=1e-6)
                signal = base_align * (1.0 + 0.5 * T_norm)
                dev = base_dev

            elif sid == S_PHANTOM:
                # Phantom: divergence (more outgoing than incoming bonds)
                w = ctx.get('w', torch.full((n,), -1.0, device=self.device))
                # Only strongly divergent oscillators (w < -1.3) get phantom affinity
                phantom_affinity = (torch.abs(w) - 1.3).clamp(0, 0.5)
                signal = base_align * (1.0 + 0.3 * phantom_affinity)
                dev = base_dev

            elif sid == S_RELATIVISTIC:
                # Relativistic: high velocity (large recent state change)
                v2c2 = ctx.get('v2c2', torch.zeros(n, device=self.device))
                signal = base_align * (1.0 + 0.5 * v2c2)
                dev = base_dev

            elif sid == S_QUANTUM:
                # Quantum: phase coherence with neighbors
                phase_coh = ctx.get('phase_coherence', torch.zeros(n, device=self.device))
                signal = base_align * (1.0 + 0.5 * phase_coh)
                dev = base_dev

            elif sid == S_MULTIVERSAL:
                # Multiversal: branch diversity (best branch alignment)
                branch_gain = ctx.get('branch_gain', torch.ones(n, device=self.device))
                signal = base_align * branch_gain
                dev = base_dev

            elif sid == S_ELEMENTAL:
                # Elemental: force coupling strength
                force_coup = ctx.get('force_coupling', torch.zeros(n, device=self.device))
                signal = base_align * (1.0 + 0.5 * force_coup)
                dev = base_dev

            elif sid == S_TEMPORAL:
                # Temporal: proper time accumulation
                t_norm = ctx.get('t_norm', torch.zeros(n, device=self.device))
                if not isinstance(t_norm, torch.Tensor):
                    t_norm = torch.full((n,), t_norm, device=self.device)
                signal = base_align * (1.0 + 0.3 * t_norm)
                dev = base_dev

            elif sid == S_CUBIT:
                # Cubit: lattice alignment
                lattice = ctx.get('lattice_align', torch.zeros(n, device=self.device))
                signal = base_align * (1.0 + 0.5 * lattice)
                dev = base_dev

            else:
                signal = base_align
                dev = base_dev

            fitness[:, k] = alpha_k * signal - lam * dev

        # Normalize fitness relative to Classical
        cls_idx = self.CANDIDATE_STATES.index(S_CLASSICAL)
        V_classical = fitness[:, cls_idx].unsqueeze(-1)
        delta_V = fitness - V_classical
        delta_V[:, cls_idx] = 0.0

        # POPULATION-BALANCE PENALTY: prevent one state from monopolizing
        # the field. Without this, VACUUM (highest α at 0.99) becomes a
        # runaway sink — once 20% of osc are quiet enough to be VACUUM,
        # they get α=0.99 boost, stay quiet, get more "stability" reward,
        # and the population collapses to all-VACUUM by step 200.
        # The penalty subtracts from each state's ΔV in proportion to how
        # over-represented it already is. Uniform share = 1/n_states.
        # State at uniform share: no penalty. State at 100% of pop: full
        # penalty (the entire field shouldn't be one state).
        n_states_active = len(self.CANDIDATE_STATES)
        uniform_share = 1.0 / n_states_active
        prev_pop_w = torch.zeros(n_states_active, device=self.device)
        for k, sid in enumerate(self.CANDIDATE_STATES):
            if sid < N_STATE_SLOTS:
                prev_pop_w[k] = self.state_weights[sid].item()
        excess = (prev_pop_w - uniform_share).clamp(min=0)  # (n_states,)
        # Penalty strength tuned to roughly cancel a 50% population state's
        # α-advantage: a state at w=0.5 gets penalty ~0.4 (which roughly
        # offsets ~0.4 of α-baseline gain).
        pop_penalty = 3.0 * excess                          # (n_states,)
        delta_V = delta_V - pop_penalty.unsqueeze(0)        # broadcast (1, n_states)

        # STOCHASTIC ROUTING with adaptive temperature.
        # Early on (low magnitudes), temperature is high → diverse states.
        # Later (high magnitudes), temperature drops → more deterministic.
        # Use ROUTABLE (interior minus locked) — locked osc keep their state.
        if routable.any():
            mean_mag = mag[routable].mean().clamp(min=1e-6)
        else:
            mean_mag = torch.tensor(1.0, device=self.device)
        # Temperature: high when mean_mag is low, low when mean_mag is high
        temperature = 0.5 / (1.0 + 5.0 * mean_mag)  # range: ~0.5 → ~0.05
        temperature = max(temperature.item(), 0.02)  # floor

        # Add per-oscillator noise for exploration (Gumbel-max trick).
        dV_int = delta_V[routable]  # (n_routable, n_states) — locked excluded
        if dV_int.shape[0] > 0:
            # Standard Gumbel-max reparameterization: sample ∝ argmax over
            # (logits + g)/T, where g ~ Gumbel(0,1). The noise MUST be scaled by
            # the same temperature as the logits. The previous form
            # (dV_int/T + 0.3·g) decoupled them, so as T fell the logits blew up
            # and the fixed-scale noise became negligible — stochastic routing
            # died abruptly instead of annealing. With proper coupling, high T =
            # noise-dominated (explore), low T = logit-dominated (exploit),
            # smoothly. (Gumbel is scale-invariant to the constant inside the
            # double-log, so no extra coefficient is needed.)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(dV_int).clamp(1e-8, 1-1e-8)))
            noisy_scores = (dV_int + gumbel_noise) / temperature

            # Sample state (Gumbel-softmax argmax)
            best_idx = noisy_scores.argmax(dim=-1)
            best_state = torch.tensor(
                [self.CANDIDATE_STATES[i] for i in best_idx],
                device=self.device, dtype=torch.long
            )
            # Write back ONLY to routable (unlocked) osc.
            # Locked osc keep their HOLOGRAPHIC assignment from consolidate_class.
            self.state_id[routable] = best_state

            # Soft probs for state weights (without noise) — also from routable only
            soft_probs = torch.softmax(dV_int / max(temperature, 0.05), dim=-1)
        else:
            soft_probs = torch.zeros(0, len(self.CANDIDATE_STATES), device=self.device)

        # Get the α for each oscillator's chosen state (reuse the cached core)
        best_a = torch.zeros(n, device=self.device)
        for k, sid in enumerate(self.CANDIDATE_STATES):
            mask = (self.state_id == sid) & self.alive_mask
            if mask.any():
                a_k, _, _, _ = alpha_for_state(
                    sid, self.theta, self.S, self.f, self.N,
                    _E_now, self.a0, ctx, core_pack=_core_pack
                )
                best_a = torch.where(mask, a_k, best_a)

        # Update state population weights — includes LOCKED osc in their
        # consolidated state. This way the population-balance penalty knows
        # the true distribution (e.g., 40 locked HOLOGRAPHIC count).
        state_weights = torch.zeros(N_STATE_SLOTS, device=self.device)
        n_int_total = int(interior.sum())
        if n_int_total > 0:
            # Count locked osc by their assigned state (which is preserved)
            for sid in self.CANDIDATE_STATES:
                if sid < N_STATE_SLOTS:
                    locked_in_state = ((self.state_id == sid) &
                                        interior & self._class_locked).sum().float()
                    state_weights[sid] = locked_in_state / n_int_total
            # Add routable contribution from soft_probs
            if soft_probs.shape[0] > 0:
                n_routable = soft_probs.shape[0]
                routable_probs = soft_probs.sum(dim=0) / n_int_total
                for k, sid in enumerate(self.CANDIDATE_STATES):
                    if sid < N_STATE_SLOTS:
                        state_weights[sid] = state_weights[sid] + routable_probs[k]
        self.state_weights = state_weights

        best_a = torch.where(self.alive_mask, best_a, torch.zeros_like(best_a))
        return s, best_a

    def _bond_inputs(self, chunk=256, split=False):
        """
        Phase-gated bond drive with P_in vs C_ij separation.
        Eq 4: E[n+1] = α·E[n] + β·P_in + Σ_j C_ij(E_i-E_j)

        Bounded to the alive high-water mark n_hi: all C rows/cols beyond
        n_hi are exactly zero (dormant slots), so skipping them is bitwise-
        identical but avoids reading the full 30 GB C tensor on every tick.
        """
        # Alive high-water mark: highest occupied index + 1. Dormant slots
        # beyond n_hi are guaranteed zero in C — safe to skip entirely.
        alive_idx = self.alive_mask.nonzero(as_tuple=True)[0]
        n_hi = int(alive_idx.max().item()) + 1 if len(alive_idx) > 0 else 1

        n = self.n_max
        cosd = self.phase_vec[:n_hi] @ self.phase_vec[:n_hi].T
        gate_full = (1.0 + cosd) * 0.5 * self.C_mask[:n_hi, :n_hi].float()

        sens = self._is_sensory[:n_hi]
        sens_f = sens.float().unsqueeze(0)
        recur_f = (~sens).float().unsqueeze(0)

        out_sens = torch.zeros(n, self.d, device=self.device)
        out_recur = torch.zeros(n, self.d, device=self.device)

        # NOTE: previously we applied self._holo_projector to drives going
        # INTO teaching slots. That was wrong — the projector is rank
        # ~n_classes in d-space, so it zeros some teach c-vector directions
        # and preserves others, causing massive class bias toward whichever
        # classes happen to align with the projector subspace.
        # The projector is still used appropriately for HOLOGRAPHIC-state
        # interior oscillators (in alpha_for_state / valence computation),
        # but NOT for the readout pathway. Sensor→teach bond drive must pass
        # through unchanged for class discrimination to work.
        for r0 in range(0, n_hi, chunk):
            r1 = min(r0 + chunk, n_hi)
            tr = torch.einsum('ijde,je->ijd', self.C[r0:r1, :n_hi], self.s[:n_hi])
            gated = tr * gate_full[r0:r1].unsqueeze(-1)
            drive_s = (gated * sens_f.unsqueeze(-1)).sum(1)
            drive_r = (gated * recur_f.unsqueeze(-1)).sum(1)

            out_sens[r0:r1] = drive_s
            out_recur[r0:r1] = drive_r
            del tr, gated, drive_s, drive_r

        if split:
            return out_sens, out_recur
        return out_sens + out_recur

    def _update_phase(self, dt=0.08):
        """Kuramoto-style phase coupling, bounded to alive high-water mark."""
        # Alive high-water mark — skip zero dormant rows/cols in C.
        alive_idx = self.alive_mask.nonzero(as_tuple=True)[0]
        n_hi = int(alive_idx.max().item()) + 1 if len(alive_idx) > 0 else 1

        phi = self.phase_angle()
        phi_hi = phi[:n_hi]
        sd = torch.sin(phi_hi.unsqueeze(0) - phi_hi.unsqueeze(1))
        bm = self.C[:n_hi, :n_hi].norm(dim=(-2, -1))
        A = self.C_mask[:n_hi, :n_hi].float() * bm
        deg = A.sum(1).clamp(min=1.0)
        # Pad coupling back to n_max so the where() broadcast works unchanged.
        coup = torch.zeros(self.n_max, device=self.device)
        coup[:n_hi] = 0.6 * (A * sd).sum(1) / deg

        omega = 2 * math.pi * self.f / 9.0 * 0.2
        free = self.alive_mask & ~self._is_sensory & ~self._is_teaching
        np_ = phi + torch.where(free, dt * (omega + coup), torch.zeros_like(phi))
        self.phase_vec = torch.stack([torch.cos(np_), torch.sin(np_)], 1)

    def _dynamics_step(self, ctx, pin_sensory=None, pin_teaching=None,
                       bond_chunk=256):
        """
        ONE PHYSICS TICK:
          0. Phase update (Kuramoto)
          1. Emergent routing (argmax valence)
          2. Bond drive (split sensory/recurrent)
          3. Direction: rotate ŝ toward d̂ by (π/2)(1−α)
          4. Energy: Eq 4 + cross-state coupling
          5. Pin sensory/teaching
          6. Update state variables
        """
        with torch.no_grad():
            alive = self.alive_mask.clone()
            n, d = self.n_max, self.d

            # 0. Phase
            self._update_phase()

            # 1. Route
            if ctx is None:
                ctx = self._build_routing_ctx()
            _, alpha = self._route_states(ctx)

            # 2. Bond drive with MULTIPLICATIVE INPUT GATING
            # Key architectural fix: the recurrent drive to each interior oscillator
            # is MODULATED by how much that oscillator's identity (c_i) resonates
            # with the current sensory input. This creates input-dependent attractors:
            # - A-resonant oscillators get amplified drive during A-input
            # - B-resonant oscillators get amplified drive during B-input
            # This is analogous to gain modulation in biological neural circuits.
            drive_sens, drive_recur = self._bond_inputs(chunk=bond_chunk, split=True)
            drive_sens = drive_sens * alive.float().unsqueeze(-1)
            drive_recur = drive_recur * alive.float().unsqueeze(-1)

            # Compute input-dependent gain for interior oscillators
            if self.input_dim is not None and pin_sensory is not None:
                # Input signature in d-space
                input_sig = pin_sensory.sum(0)  # (d,) sum of sensory state vectors
                input_sig = input_sig / input_sig.norm().clamp(min=1e-6)
                # Resonance: how aligned is each oscillator's c with input?
                resonance = (self.c * input_sig.unsqueeze(0)).sum(-1)  # (n,)
                # Gain: resonant oscillators get amplified recurrent drive
                # gain ∈ [0.1, 2.0]: non-resonant get suppressed, resonant get boosted
                gain = 0.1 + 1.9 * ((resonance + 1.0) / 2.0)  # map [-1,1] to [0.1, 2.0]
                # Only gate interior oscillators (not sensory/teaching)
                interior_mask = ~self._is_sensory & ~self._is_teaching
                gain = torch.where(interior_mask, gain, torch.ones_like(gain))
                # Apply gain to recurrent drive
                drive_recur = drive_recur * gain.unsqueeze(-1)

            drive = drive_sens + drive_recur

            d_mag = drive.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            d_hat = drive / d_mag

            # 3. Direction: rotate ŝ toward d̂
            s = self.s
            s_mag = s.norm(dim=-1, keepdim=True)
            s_hat = torch.where(s_mag < 1e-6, d_hat, s / s_mag.clamp(min=1e-8))

            perp = d_hat - (d_hat * s_hat).sum(-1, keepdim=True) * s_hat
            pn = perp.norm(dim=-1, keepdim=True)
            perp_hat = torch.where(pn > 1e-6, perp / pn.clamp(min=1e-8),
                                   torch.zeros_like(perp))

            theta_ev = (math.pi / 2) * (1.0 - alpha).clamp(0, 1).unsqueeze(-1)
            new_hat = torch.cos(theta_ev) * s_hat + torch.sin(theta_ev) * perp_hat

            # 4. Energy — Eq 4 + cross-state coupling
            E = (s_mag.squeeze(-1) ** 2)
            beta = 0.5  # Increased from 0.15: sensory input must be strong enough to steer dynamics
            sens_mag = drive_sens.norm(dim=-1)
            P_in = sens_mag ** 2

            # TEACHING READOUT: Use ONLY sensory drive for teaching P_in.
            #
            # WHY: Interior oscillators converge to the same attractor regardless
            # of input (recurrent bonds dominate). So drive_recur is input-INDEPENDENT.
            # But drive_sens comes from sensor→teaching bonds, which encode the
            # input pattern directly. Using only drive_sens makes the readout
            # input-dependent, enabling discrimination.
            #
            # The signed projection onto c_i gives class-specific activation:
            # - Bonds formed during class A training: C[teach0, sensor_j] = outer(c0, s_j_A)
            # - During eval with A-input: drive_sens · c0 = sum_j (s_j_A_train · s_j_A_eval) > 0
            # - During eval with B-input: drive_sens · c0 = sum_j (s_j_A_train · s_j_B_eval) ≈ 0
            #   (because A and B inputs are different patterns)
            # This creates natural discrimination WITHOUT forgetting.
            teach_mask = self._is_teaching
            # For teaching: use sensory drive only (input-dependent)
            drive_sens_proj = (drive_sens * self.c).sum(-1)  # (n,)
            P_in_teach = drive_sens_proj.clamp(min=0.0) ** 2
            # Regular oscillators use full sensory drive norm
            P_in = torch.where(teach_mask, P_in_teach, P_in)

            # Consciousness Eq 2: P_in*(1+γ_attention*A(t)) for attended oscillators
            A_att = ctx.get('A', 0.5)
            A_val = float(A_att.item() if isinstance(A_att, torch.Tensor) else A_att)
            consciousness_mask = (self.state_id == S_CONSCIOUSNESS) & ~teach_mask
            P_in = torch.where(consciousness_mask, P_in * (1.0 + 0.5 * A_val), P_in)

            # Teaching slots need a base alpha (moderate retention).
            _teach_alpha = 0.5 if self.fixes.get('fix2') else 0.3
            alpha = torch.where(teach_mask,
                                torch.full_like(alpha, _teach_alpha),
                                alpha)

            # Cross-state energy flow (Unified Master Equation)
            mean_E_per_state = torch.zeros(N_STATE_SLOTS, device=self.device)
            for sid in self.CANDIDATE_STATES:
                mask_sid = (self.state_id == sid) & alive
                if mask_sid.any():
                    mean_E_per_state[sid] = E[mask_sid].mean()

            sid_per_osc = self.state_id
            C_rows = self.C_states[sid_per_osc]
            E_diff = mean_E_per_state.unsqueeze(0) - E.unsqueeze(-1)
            w_states = self.state_weights[:N_STATE_SLOTS].unsqueeze(0)
            cross_flow = (C_rows * E_diff * w_states).sum(-1)
            # Teaching slots don't participate in cross-state flow
            cross_flow = torch.where(teach_mask, torch.zeros_like(cross_flow), cross_flow)

            # Eq 4: E[n+1] = α·E[n] + β·P_in + cross_flow
            E_next = alpha * E + beta * P_in + cross_flow

            # Vacuum zero-point floor
            vacuum_mask = (sid_per_osc == S_VACUUM)
            E_next = torch.where(vacuum_mask,
                                 torch.maximum(E_next, self.vacuum_energy * 0.3),
                                 E_next)

            # Time dilation: high-γ oscillators change slower
            # CRITICAL: Teaching slots BYPASS time dilation.
            # They are readout nodes that must respond immediately to bond drive.
            dilation = 1.0 / self.gamma.clamp(min=1.0)
            # Teaching slots get dilation=1.0 (no slowdown)
            dilation = torch.where(teach_mask, torch.ones_like(dilation), dilation)
            E_next = E + (E_next - E) * dilation

            # Phantom energy GAIN is already applied by Eq 4 above: for Phantom
            # nodes, alpha sits in the [1.00, 1.05] band (the only state with
            # α>1), so E_next = alpha·E already grows their energy by the
            # framework-correct amount. A second multiplier here would apply the
            # gain TWICE per tick and break the preservation hierarchy. Removed.

            # Clamp energy to prevent explosion
            # Teaching slots get higher clamp to preserve discriminative signal
            max_E = torch.where(teach_mask,
                                torch.full_like(E_next, 50.0),
                                torch.full_like(E_next, 5.0))
            E_next = E_next.clamp(min=0.0)
            E_next = torch.min(E_next, max_E)

            # Assemble new state vector
            new_mag = torch.sqrt(E_next.clamp(min=0.0))
            new_s = new_hat * new_mag.unsqueeze(-1)

            # 5. Pin sensory and teaching
            if pin_sensory is not None:
                new_s[:self.input_dim] = pin_sensory
            if pin_teaching is not None and self.t_start is not None:
                new_s[self.t_start:self.t_end] = pin_teaching

            # UERF PRESERVATION (framework fidelity, not an optimization):
            # Class-locked oscillators are the permanent reference nodes —
            # α≈0.998→1.0, the information-preservation regime. Physically α→1
            # means state is CONSERVED across a tick, not re-derived from drive.
            # Only _class_locked is fully frozen here. Crystallized nodes sit at
            # α≈0.98 (minimal decay, still slowly evolving) and are handled by
            # the decay block — freezing them here would over-preserve and
            # contradict their α<1. _class_locked is always interior (see
            # consolidate_class), so this never clobbers the sensory/teaching
            # pins applied above.
            if self._class_locked.any():
                new_s = torch.where(self._class_locked.unsqueeze(-1),
                                    self.s, new_s)

            self.s = new_s * alive.float().unsqueeze(-1)

            # 6. Update state variables
            self._update_state_variables(alive)

    def _update_state_variables(self, alive):
        """Evolve auxiliary state variables each tick."""
        with torch.no_grad():
            s = self.s
            mag = s.norm(dim=-1)
            E = mag ** 2

            # Local temperature (energy variance over history)
            self._E_history = torch.roll(self._E_history, 1, dims=1)
            self._E_history[:, 0] = E
            self.local_T = self._E_history.var(dim=1).clamp(min=1e-3, max=2.0)

            # Only genuinely divergent oscillators should enter PHANTOM.
            out_flow = self.C_mask.float().sum(dim=1)
            in_flow = self.C_mask.float().sum(dim=0)
            # Normalized divergence: positive = more outgoing
            div_norm = (out_flow - in_flow) / (out_flow + in_flow + 1.0).clamp(min=1.0)
            # EMA update: slow adaptation prevents momentary spikes from triggering PHANTOM
            target_w = -1.0 - 0.3 * torch.tanh(div_norm * 2.0)  # range [-1.3, -0.7]
            self.local_w = 0.95 * self.local_w + 0.05 * target_w
            # Clamp to prevent extreme values
            self.local_w = self.local_w.clamp(-1.5, -0.7)

            # Lorentz γ from full vector velocity
            if self._prev_s is not None:
                ds = s - self._prev_s
                v_full = ds.norm(dim=-1)
                scale = mag[alive].mean().clamp(min=0.1) if alive.any() else torch.tensor(0.1, device=self.device)
                beta = (v_full / scale).clamp(0.0, 0.95)
                self.local_v2c2 = (beta ** 2).clamp(0.0, 0.95)
                self.gamma = (1.0 / torch.sqrt(1.0 - self.local_v2c2)).clamp(1.0, 3.0)
            # Proper time accrual with dilation
            dt = 1.0
            activity = (mag > 0.05).float()
            self.proper_time = self.proper_time + dt * activity / self.gamma

            # Fractal scale memory (sample at φ^k intervals)
            ec = self.experience_count
            self.scale_memory[:, 0] = s
            for k in range(1, 4):
                period = max(1, int(round(PHI ** k)))
                if ec % period == 0:
                    self.scale_memory[:, k] = self.scale_memory[:, k - 1]

            # Multiversal branches
            for j in range(4):
                T_self = 0.3 + 0.1 * j
                self.s_branches[:, j] = (
                    (1.0 - T_self) * self.s_branches[:, j] + T_self * s
                )
                # Decorrelate from other branches
                for k in range(4):
                    if k != j:
                        overlap = (self.s_branches[:, j] * self.s_branches[:, k]).sum(-1, keepdim=True)
                        sk_mag2 = (self.s_branches[:, k] ** 2).sum(-1, keepdim=True).clamp(min=1e-8)
                        self.s_branches[:, j] = self.s_branches[:, j] - 0.03 * overlap / sk_mag2 * self.s_branches[:, k]

            # Branch weights
            s_mag_local = mag.clamp(min=1e-8)
            branch_aligns = torch.zeros(self.n_max, 4, device=self.device)
            for j in range(4):
                sb = self.s_branches[:, j]
                sb_mag = sb.norm(dim=-1).clamp(min=1e-8)
                pred_match = (sb * s).sum(-1) / (sb_mag * s_mag_local)
                branch_aligns[:, j] = pred_match.clamp(-1, 1)
            bw_new = torch.softmax(branch_aligns * 3.0, dim=-1)
            self.branch_weights = 0.9 * self.branch_weights + 0.1 * bw_new

            # Store current prediction before updating (for accuracy measurement)
            self._prev_prediction = self.future_prediction.clone()
            # Predict: current state + velocity (momentum-based prediction)
            if self._prev_s is not None:
                ds = s - self._prev_s
                self.future_prediction = s + ds * self.gamma.unsqueeze(-1) * 0.5
            else:
                self.future_prediction = s.clone()

            # Force channels (running coupling constants)
            Q2 = E.clamp(min=1e-3)
            self.force_channels[:, 0] = (0.07 + 0.02 * torch.log(1.0 + Q2)).clamp(0, 1)
            self.force_channels[:, 1] = (1.0 / (1.0 + 0.3 * torch.log(1.0 + Q2))).clamp(0, 1)
            self.force_channels[:, 2] = (0.03 + 0.05 * torch.tanh(Q2 / 3.0)).clamp(0, 1)
            self.force_channels[:, 3] = (Q2 / (1.0 + Q2)).clamp(0, 1)

            # Vacuum energy (inverse of activity)
            quiet_proxy = torch.exp(-mag * 2.0)
            self.vacuum_energy = 0.95 * self.vacuum_energy + 0.05 * quiet_proxy

            self._prev_mag = mag.clone()
            self._prev_s = s.clone()


# Attach dynamics methods to UERFField
for _m in ('_build_routing_ctx', '_route_states', '_bond_inputs', '_update_phase',
           '_dynamics_step', '_update_state_variables'):
    setattr(UERFField, _m, getattr(UERFDynamics, _m))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — VALENCE-DRIVEN LEARNING + GROWTH + DEATH
# ═══════════════════════════════════════════════════════════════════════════════
class UERFLearning:
    """Mixin: learning, crystallization, birth/death."""

    def _valence_per_osc(self, lam=0.25):
        """V_i = α_i⟨s_i,R_i⟩ − λ‖s_i−I_i‖²"""
        alpha = self._alpha_of_current_state({})
        R = self.c
        I = self.c * self.state_magnitude().unsqueeze(-1)
        return valence(self.s, R, I, alpha, lam=lam), alpha

    def _alpha_of_current_state(self, ctx):
        """α for each oscillator under its currently routed state."""
        n = self.n_max
        out = torch.zeros(n, device=self.device)
        # alpha_core is state-independent — one evaluation serves every state
        core_pack = alpha_core(self.theta, self.S, self.f, self.N, self.a0)
        E_now = self.energy()
        for sid in self.CANDIDATE_STATES:
            m = (self.state_id == sid)
            if not m.any():
                continue
            a, _, _, _ = alpha_for_state(sid, self.theta, self.S, self.f,
                                         self.N, E_now, self.a0, ctx,
                                         core_pack=core_pack)
            out = torch.where(m, a, out)
        return out

    def _learn_bonds(self, lr=0.08, decay=1e-3, lam=0.25, chunk=128):
        """
        VALENCE-ASCENT BOND UPDATE.
        ΔC[i,j] ∝ α_i · (R_i − λ(s_i−I_i)) ⊗ s_j  (outer product)
        
        THREE PATHWAYS:
          1. Interior ← (Interior + Sensor): interior oscillators learn from all alive
          2. Teaching ← Interior: readout bonds (how interior drives teaching slots)
          3. Interior ← Teaching: feedback bonds (teaching signal back-propagates)
        """
        with torch.no_grad():
            alive = self.alive_mask
            interior = alive & ~self._is_sensory & ~self._is_teaching
            teach = self._is_teaching
            sens = self._is_sensory
            if not interior.any():
                return

            alpha = self._alpha_of_current_state({})
            s = self.s
            c = self.c
            mag = self.state_magnitude()
            I = c * mag.unsqueeze(-1)

            # Gradient of valence w.r.t. s_i direction
            grad_v = alpha.unsqueeze(-1) * c - 2 * lam * (s - I)

            n = self.n_max
            d = self.d

            # PATHWAY 1: Interior receivers learn from all alive senders.
            # EXCEPT class-locked osc — their incoming bonds are FROZEN
            # (consolidated class memory). They still PARTICIPATE as senders
            # (their pattern drives other osc), but they no longer learn.
            #
            # Bounded to the alive high-water mark n_hi: dormant senders have
            # s=0 (zero contribution) and dormant receivers are masked out, so
            # restricting the outer product and the writes to [:n_hi] is
            # bitwise-identical while skipping the dormant-slot bandwidth.
            alive_hw = alive.nonzero(as_tuple=True)[0]
            n_hi = int(alive_hw.max().item()) + 1 if len(alive_hw) > 0 else 1
            receivers = interior & ~self._class_locked
            for r0 in range(0, n_hi, chunk):
                r1 = min(r0 + chunk, n_hi)
                recv_mask = receivers[r0:r1]
                if not recv_mask.any():
                    continue

                gv = grad_v[r0:r1]  # (chunk, d)
                delta_C = torch.einsum('id,je->ijde', gv, s[:n_hi])  # (chunk, n_hi, d, d)

                sender_alive = alive[:n_hi].float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                delta_C = delta_C * sender_alive

                alpha_scale = alpha[r0:r1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                self.C[r0:r1, :n_hi] += lr * alpha_scale * delta_C * recv_mask.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Decay bonds. Locked rows are always frozen (no decay), and so
                # are TEACHING rows: teach slots are not Pathway-1 receivers,
                # but they share chunks with interior receivers, so the old
                # blanket decay multiplied the interior→teach readout bonds by
                # (1-decay) every step — 0.999^10000 ≈ 4.5e-5 — silently
                # erasing exactly the bonds consolidate_class ranks on.
                locked_chunk = self._class_locked[r0:r1]
                teach_chunk = self._is_teaching[r0:r1]
                if self.fixes.get('fix6'):
                    is_receiver = recv_mask & ~locked_chunk
                    decay_per_row = torch.where(
                        is_receiver,
                        torch.tensor(1.0 - decay, device=self.device),
                        torch.tensor(1.0, device=self.device))   # non-receivers frozen
                else:
                    decay_per_row = torch.where(
                        locked_chunk | teach_chunk,
                        torch.tensor(1.0, device=self.device),     # locked/teach: no decay
                        torch.tensor(1.0 - decay, device=self.device))
                self.C[r0:r1, :n_hi] *= decay_per_row.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                del delta_C

            # PATHWAY 2: Teaching slots RECEIVE from interior (readout pathway)
            # CONTRASTIVE LEARNING:
            # - The ACTIVE class slot (pinned during training) gets bonds STRENGTHENED
            #   so that the current interior pattern drives it.
            # - INACTIVE class slots get bonds WEAKENED (anti-Hebbian)
            #   so they don't activate for this pattern.
            # This creates discriminative readout.
            if self.t_start is not None:
                t0, t1 = self.t_start, self.t_end
                n_cls = t1 - t0
                teach_c = c[t0:t1]  # (n_classes, d)

                # Determine which class is currently active (highest pinned magnitude)
                teach_mags = s[t0:t1].norm(dim=-1)
                active_class = teach_mags.argmax().item()

                # Interior oscillators that will form readout bonds
                interior_idx = interior.nonzero(as_tuple=True)[0]
                # Use top-k most active interior oscillators for efficiency
                int_mags = mag[interior_idx]
                n_bond_targets = min(40, len(interior_idx))
                if len(interior_idx) > n_bond_targets:
                    _, top_k = int_mags.topk(n_bond_targets)
                    bond_idx = interior_idx[top_k]
                else:
                    bond_idx = interior_idx

                if len(bond_idx) > 0:
                    # SENSORY-BASED READOUT — TRUE RUNNING MEAN (not EMA).
                    #
                    # Bonds for class c are the MEAN outer-product over all
                    # examples of class c. Phase B examples for classes 5-9
                    # only update the accumulators for classes 5-9. Phase A
                    # bonds (classes 0-4) never decay or drift — they stay
                    # as the mean of Phase A inputs forever.
                    #
                    # This solves catastrophic forgetting at the bond level:
                    # the bonds for different classes are stored in disjoint
                    # accumulator slices and never overwrite each other.
                    sensor_idx = sens.nonzero(as_tuple=True)[0]
                    if self._teach_bond_sum is not None and len(sensor_idx) > 0:
                        self._teach_bond_count[active_class] += 1
                        count = self._teach_bond_count[active_class].item()
                        target_dir = teach_c[active_class]
                        # Vectorized accumulator update for all sensors at once
                        s_sensors = s[sensor_idx]                   # (n_sens, d)
                        s_norms = s_sensors.norm(dim=-1)
                        valid = s_norms > 1e-6
                        if valid.any():
                            # outer(target_dir, s_j) for each sensor j
                            outers = torch.einsum('d,je->jde',
                                                  target_dir, s_sensors)  # (n_sens, d, d)
                            self._teach_bond_sum[active_class, sensor_idx] += outers
                        # Write the mean to actual bond matrix
                        # (do this every step, cheap relative to dynamics)
                        if count >= 1:
                            mean_bonds = (self._teach_bond_sum[active_class, sensor_idx]
                                          / count)
                            self.C[t0 + active_class, sensor_idx] = mean_bonds

                    # Auxiliary interior→teaching bonds (smaller, additive).
                    # via += over thousands of steps, eventually dominating the
                    # clean normalized sensor→teach mean-accumulator bonds and
                    # polluting the discriminative readout. When fix3 is on we
                    # rely solely on the sensor-based mean bonds above.
                    if not self.fixes.get('fix3'):
                        target_dir = teach_c[active_class]
                        for j_idx in bond_idx[:20]:
                            s_j = s[j_idx]
                            if s_j.norm() > 1e-6:
                                self.C[t0 + active_class, j_idx] += (
                                    0.02 * torch.outer(target_dir, s_j))
                    # Inactive classes: completely untouched.

            # PATHWAY 3: Sensor → Interior bonds (input pathway)
            # Sensors are pinned, so their bonds TO interior should strengthen
            # when the interior oscillator's valence benefits from sensor input.
            # This happens naturally in Pathway 1 since sensors are alive senders.

            # VECTORIZED over ALL interior nodes (was capped at first 50, which
            # silently left every node beyond 50 — i.e. nearly the whole field at
            # n_max=1800 with growth — without the self-coupling that reshapes the
            # energy landscape and confines dynamics against catastrophic
            # forgetting). The physics must scale with the field. Bitwise
            # identical to the per-node loop, applied to the full interior set.
            _autapse_valid = interior & (mag > 1e-6)
            _av_idx = _autapse_valid.nonzero(as_tuple=True)[0]
            if len(_av_idx) > 0:
                _s_v = s[_av_idx]                                   # (k, d)
                _outers = torch.einsum('kd,ke->kde', _s_v, _s_v)    # (k, d, d)
                _denom = (mag[_av_idx] ** 2 + 1e-8).view(-1, 1, 1)
                self.C[_av_idx, _av_idx] += lr * 0.1 * _outers / _denom
                self.C[_av_idx, _av_idx] *= (1.0 - decay * 0.5)

            # Prune weak bonds, enforce budget
            self._prune_bonds()

    def _prune_bonds(self):
        """Prune weak bonds and enforce per-oscillator budget.

        Bounded to the alive high-water mark n_hi: all C rows/cols beyond
        n_hi are exactly zero (dormant slots), so restricting the norm scan,
        weak-kill, budget rescale, and mask rebuild to [:n_hi, :n_hi] is
        bitwise-identical but stops reading/writing the full O(n_max²·d²)
        tensor twice per learning step (the single biggest per-step cost
        after the bond drive itself)."""
        with torch.no_grad():
            n = self.n_max
            alive = self.alive_mask
            sens = self._is_sensory
            teach = self._is_teaching
            cr = self.crystallized()
            locked = self._class_locked

            alive_idx = alive.nonzero(as_tuple=True)[0]
            n_hi = int(alive_idx.max().item()) + 1 if len(alive_idx) > 0 else 1
            C_hi = self.C[:n_hi, :n_hi]            # view — writes hit self.C

            # Norm of each bond (occupied block only)
            tn = C_hi.norm(dim=(-2, -1))           # (n_hi, n_hi)

            # Kill very weak bonds — EXCEPT bonds touching locked osc
            # (those represent consolidated class memory; never erase them).
            weak = tn < 0.02
            locked_rows = locked[:n_hi].unsqueeze(1).expand(n_hi, n_hi)
            locked_cols = locked[:n_hi].unsqueeze(0).expand(n_hi, n_hi)
            # Also protect teaching-slot rows from weak pruning so that
            # the readout bonds C[teach_c, sensor] persist even if some
            # individual sensor bonds are below threshold.
            teach_rows = teach[:n_hi].unsqueeze(1).expand(n_hi, n_hi)
            weak = weak & ~(locked_rows | locked_cols | teach_rows)
            C_hi[weak] = 0.0

            # Budget: limit total incoming bond strength per oscillator
            recv_total = torch.zeros(n, device=self.device)
            recv_total[:n_hi] = tn.sum(dim=1)
            state_alpha = self._alpha_of_current_state({})
            n_alive = alive.sum().clamp(min=1).float()
            budget_per_osc = 0.5 * state_alpha * n_alive

            # Teaching slots get extra budget (readout must persist)
            budget_per_osc = torch.where(teach, budget_per_osc * 3.0, budget_per_osc)
            # Crystallized: 2x
            budget_per_osc = torch.where(cr, budget_per_osc * 2.0, budget_per_osc)
            # Class-locked: 10x (effectively unbounded — consolidated memory)
            budget_per_osc = torch.where(locked, budget_per_osc * 10.0, budget_per_osc)

            over_budget = (recv_total > budget_per_osc) & alive
            if over_budget.any():
                scale_factor = budget_per_osc / recv_total.clamp(min=1e-6)
                scale_factor = torch.where(over_budget, scale_factor,
                                           torch.ones_like(scale_factor))
                C_hi.mul_(scale_factor[:n_hi].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

            # Rebuild mask (rows/cols ≥ n_hi are zero ⇒ mask False, identical)
            tn = C_hi.norm(dim=(-2, -1))
            self.C_mask = torch.zeros(n, n, dtype=torch.bool, device=self.device)
            self.C_mask[:n_hi, :n_hi] = tn > 0.02
            # MASK SEMANTICS: C_mask[i, j] = True means "i receives from j"
            # (because C[i,j] @ s[j] contributes to drive of oscillator i)
            #
            # Sensors don't RECEIVE (they're pinned inputs)
            self.C_mask[sens, :] = False
            # Teaching doesn't SEND (nobody receives FROM teaching)
            # C_mask[:, teach] = False means no oscillator receives from teaching
            self.C_mask[:, teach] = False
            # Teaching DOES receive (interior -> teaching is the readout pathway)
            # C_mask[teach, :] stays as-is (determined by bond strength > threshold)

    def _competitive_c_learning(self, input_sig=None, lr_c=0.01):
        """
        Competitive learning for c_i vectors: active oscillators drift their
        identity toward the current input signature, making them specialize.
        This creates input-dependent resonance for the gain modulation.
        """
        if input_sig is None:
            return
        with torch.no_grad():
            interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
            if not interior.any():
                return
            cr = self.crystallized()
            # Only non-crystallized, non-locked interior osc can adapt
            plastic = interior & ~cr & ~self._class_locked
            if not plastic.any():
                return
            
            # Find the top-k most active oscillators (winners)
            mag = self.state_magnitude()
            plastic_idx = plastic.nonzero(as_tuple=True)[0]
            plastic_mags = mag[plastic_idx]
            n_winners = min(10, len(plastic_idx))  # top 10 winners adapt
            if n_winners == 0:
                return
            _, top_k = plastic_mags.topk(n_winners)
            winner_idx = plastic_idx[top_k]
            
            # Winners drift c_i toward input signature
            input_dir = input_sig / input_sig.norm().clamp(min=1e-6)
            for idx in winner_idx:
                # Blend c_i toward input direction (small step)
                new_c = (1.0 - lr_c) * self.c[idx] + lr_c * input_dir
                self.c[idx] = new_c / new_c.norm().clamp(min=1e-6)

    def _identity_drift(self, lr=0.015):
        """
        FIX 3: Crystallization requires SUSTAINED high valence, not just drift.
        Only oscillators with consistently high valence drift toward golden.
        """
        with torch.no_grad():
            V, _ = self._valence_per_osc()
            interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
            cr = self.crystallized()
            target = interior & ~cr & ~self._class_locked

            if not target.any():
                return

            # Update valence accumulator (running average)
            self._valence_accumulator[target] = (
                0.95 * self._valence_accumulator[target] + 0.05 * V[target]
            )
            self._valence_count[target] += 1

            # Only drift oscillators with ABOVE-THRESHOLD sustained valence.
            # The accumulator is already an EMA-mean — do NOT divide by the
            # update count (that drove it to ~0 and killed all drift).
            mean_val = self._valence_accumulator
            high_valence = (mean_val > self._crystallization_threshold * 0.5) & target

            if not high_valence.any():
                return

            # Drift strength proportional to valence (normalized)
            v = V[high_valence].clone()
            v_range = v.max() - v.min()
            if v_range > 1e-6:
                v = (v - v.min()) / v_range
            else:
                v = torch.ones_like(v) * 0.5

            pull = lr * v
            hv_idx = high_valence.nonzero(as_tuple=True)[0]
            self.theta[hv_idx] = self.theta[hv_idx] + pull * (THETA_G - self.theta[hv_idx])
            self.S[hv_idx] = self.S[hv_idx] + pull * (S_PHI - self.S[hv_idx])

            # Check for new crystallizations
            newly = self.crystallized() & target
            if newly.any():
                self.crystallization_events += int(newly.sum())

    def _contradiction(self):
        """
        Field contradiction signal — drives Phantom birth and emergent growth.
        Returns value in [0, 1].

        Two sub-signals, combined as max:
          1. Low mean valence (field can't satisfy current pattern)
          2. High valence variance (osc disagree about the pattern)

        Note: previously included a teach-slot magnitude sub-signal, but
        teach magnitudes are only pinned DURING the relaxation loop. By
        the time _contradiction() is called (after experience() returns),
        teach magnitudes have decayed back to ~0, making that signal a
        useless constant ~0.98. Removed.
        """
        V, _ = self._valence_per_osc()
        interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
        if int(interior.sum()) < 3:
            return 0.0
        vi = V[interior]

        # 1. Mean valence — V is in roughly [-1, 1]; map to [0, 1] where
        #    1.0 = max contradiction (very negative V), 0.0 = no contradiction.
        #    A well-trained field should have V slightly positive → mean_c ~ 0.4.
        #    A struggling field has V near 0 or negative → mean_c ~ 0.5-1.0.
        mean_c = float((1.0 - vi.mean().clamp(-1, 1)) / 2.0)

        # 2. Variance — high std means osc are at very different operating
        #    points (the field hasn't found consensus). Scale so std=0 → 0,
        #    std=0.5 → 1.0 (anything above 0.5 std means severe disagreement).
        var_c = float((vi.std() * 2.0).clamp(0, 1))

        return max(mean_c, var_c)

    def _phantom_birth(self, n_spawn, target_teach=None):
        """
        Spawn PHANTOM oscillators into dormant slots.
        FIX 5: More aggressive spawning with better initialization.
        """
        with torch.no_grad():
            dormant = (~self.alive_mask & ~self._is_teaching).nonzero(as_tuple=True)[0]
            if len(dormant) == 0:
                return 0
            k = min(n_spawn, len(dormant))
            slots = dormant[:k]

            # Initialize near golden ratio for high base α
            self.theta[slots] = THETA_G + torch.randn(k, device=self.device) * 0.3
            self.S[slots]     = S_PHI + torch.randn(k, device=self.device) * 0.2
            self.f[slots]     = torch.rand(k, device=self.device) * 6.0 + 3.0  # near harmonics
            self.N[slots]     = torch.rand(k, device=self.device) * 6.0 + 3.0
            self.a0[slots]    = torch.rand(k, device=self.device) * 0.05 + 0.92
            self.age[slots]   = 0

            # Initialize state with small random + bias toward active region
            interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
            if interior.any():
                mean_s = self.s[interior].mean(0)
                init_s = mean_s.unsqueeze(0) * 0.3 + torch.randn(k, self.d, device=self.device) * 0.1
            else:
                init_s = torch.randn(k, self.d, device=self.device) * 0.1

            self.s[slots] = init_s

            # Coherence direction
            cr = torch.randn(k, self.d, device=self.device)
            self.c[slots] = cr / cr.norm(dim=1, keepdim=True).clamp(min=1e-8)

            # Connect sparsely to existing interior oscillators
            active_interior = interior.nonzero(as_tuple=True)[0]
            if len(active_interior) > 0:
                n_connect = min(5, len(active_interior))
                for s_idx in slots:
                    targets = active_interior[torch.randperm(len(active_interior))[:n_connect]]
                    for t_idx in targets:
                        # Small random bond
                        self.C[s_idx, t_idx] = torch.randn(self.d, self.d, device=self.device) * 0.03
                        self.C[t_idx, s_idx] = torch.randn(self.d, self.d, device=self.device) * 0.03

            # Connect to teaching slot if specified
            if target_teach is not None:
                for s_idx in slots:
                    self.C[target_teach, s_idx] = torch.randn(self.d, self.d, device=self.device) * 0.05

            self.alive_mask[slots] = True
            self.state_id[slots] = S_PHANTOM
            self.births += k

            # Update bond mask
            tn = self.C.norm(dim=(-2, -1))
            self.C_mask = tn > 0.02

            return k

    def _age_and_cull(self):
        """Age oscillators and cull low-energy ones."""
        with torch.no_grad():
            interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
            cr = self.crystallized()
            self.age[interior] += 1

            # Cull: old + low energy + not crystallized + NOT class-locked
            E = self.energy()
            old = self.age > 200
            low_E = E < 0.001
            cullable = interior & old & low_E & ~cr & ~self._class_locked

            if cullable.any():
                self.alive_mask[cullable] = False
                self.C[cullable] = 0.0
                self.C[:, cullable] = 0.0
                self.C_mask[cullable] = False
                self.C_mask[:, cullable] = False
                self.deaths += int(cullable.sum())

    def consolidate_class(self, class_id, n_lock=8):
        """
        Freeze top-n_lock interior oscillators that drive teaching slot
        class_id into the highest-α state (HOLOGRAPHIC) + crystallized.
        After consolidation, these oscillators are permanent reference
        nodes for class_id — they never decay, never compete, never die.

        Call this after each phase of training to lock in class memory.
        """
        with torch.no_grad():
            if self.t_start is None or class_id >= self.n_classes:
                return 0
            teach_idx = self.t_start + class_id
            interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
            if not interior.any():
                return 0

            # Bond strength from each interior osc into this class's teach slot
            strengths = self.C[teach_idx].norm(dim=(-2, -1))   # (n_max,)

            # FALLBACK: if no interior→teach bonds exist (checkpoints trained
            # before the Pathway-1 teach-row decay fix, or fix3 sensor-only
            # readout), rank by how well each interior identity vector aligns
            # with this class's mean input signature, reconstructed from the
            # sensor→teach mean accumulator: _teach_bond_sum[k, j] =
            # count·outer(c_teach_k, mean_s_j), so c_teach_k @ sum gives
            # count·mean_s_j, and the class signature is Σ_j mean_s_j.
            if (float(strengths[interior].max()) < 1e-4
                    and self._teach_bond_sum is not None
                    and float(self._teach_bond_count[class_id]) > 0):
                sensor_idx = self._is_sensory.nonzero(as_tuple=True)[0]
                if len(sensor_idx) > 0:
                    cnt = self._teach_bond_count[class_id].clamp(min=1)
                    c_k = self.c[teach_idx]
                    mean_s = torch.einsum(
                        'd,jde->je', c_k,
                        self._teach_bond_sum[class_id, sensor_idx]) / cnt
                    sig = mean_s.sum(0)
                    sig_n = sig.norm().clamp(min=1e-6)
                    if float(sig_n) > 1e-6:
                        sig = sig / sig_n
                        strengths = (self.c * sig.unsqueeze(0)).sum(-1).clamp(min=0)

            # Already-locked candidates excluded; also exclude crystallized
            # locked-to-other-class to avoid stealing
            candidates = interior & ~self._class_locked
            if not candidates.any():
                return 0
            cand_strengths = torch.where(candidates, strengths,
                                          torch.zeros_like(strengths))
            n_take = min(n_lock, int(candidates.sum().item()))
            if n_take == 0:
                return 0
            _, top_indices = cand_strengths.topk(n_take)

            locked_count = 0
            for idx in top_indices:
                if cand_strengths[idx] < 1e-4:
                    continue                # don't lock osc with no real bond
                # Declare this oscillator a permanent reference node for the
                # class. It moves to the highest-preservation state and is
                # frozen via _class_locked (which every decay/dynamics path
                # already honors). We do NOT back-date the valence accumulator
                # to fake crystallized() — locking is what preserves it, and
                # crystallized() stays an honest measure of earned valence.
                self.theta[idx] = THETA_G
                self.S[idx] = S_PHI
                self.state_id[idx] = S_HOLOGRAPHIC
                self._class_locked[idx] = True
                self._locked_class[idx] = class_id
                self.crystallization_events += 1
                locked_count += 1
            return locked_count

    def grow_capacity(self, n_grow=200):
        """
        EMERGENT NEUROGENESIS at the tensor level.

        When dormant slots are exhausted AND contradiction stays high, the
        field itself signals it needs more substrate. This method extends
        every per-oscillator tensor by n_grow new slots, all marked dormant
        (alive_mask=False) and ready for _phantom_birth to populate.

        This is the analog of biological neurogenesis: when the existing
        cellular substrate cannot accommodate a new pattern, new cells
        are generated. Memory of existing patterns is preserved exactly
        because we EXTEND tensors (concatenate new zeros), never reindex.

        Returns the new n_max after growth.

        Memory cost: bond tensor C grows as O((n_max + n_grow)²). At
        d=32, growing from 400 → 600 adds ~330 MB.
        """
        with torch.no_grad():
            dev = self.device
            old_n = self.n_max
            new_n = old_n + n_grow
            d = self.d

            # ── 1-D per-osc tensors (most attributes)
            scalar_attrs_zero = [
                'age', 'proper_time', 'local_v2c2', '_valence_accumulator',
                '_valence_count',
            ]
            scalar_attrs_one = ['gamma']
            scalar_attrs_half = ['local_T']  # init 0.5
            scalar_attrs_neg1 = ['local_w']  # init -1.0
            scalar_attrs_0_1 = ['vacuum_energy']  # init 0.1
            bool_attrs = ['alive_mask', '_is_sensory', '_is_teaching', '_class_locked']
            long_attrs_neg1 = ['_locked_class']
            long_attrs_classical = ['state_id']

            # Random-init scalars (need same distribution as init)
            random_attrs = {
                'theta':  ('randn', 0.8, THETA_G),
                'S':      ('randn', 0.5, S_PHI),
                'f':      ('rand',  8.0, 1.0),
                'N':      ('rand',  8.0, 1.0),
                'a0':     ('rand',  0.15, 0.85),
            }

            for attr in scalar_attrs_zero:
                if not hasattr(self, attr):
                    continue
                cur = getattr(self, attr)
                if cur is None:
                    continue
                add = torch.zeros(n_grow, dtype=cur.dtype, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in scalar_attrs_one:
                cur = getattr(self, attr)
                add = torch.ones(n_grow, dtype=cur.dtype, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in scalar_attrs_half:
                cur = getattr(self, attr)
                add = torch.full((n_grow,), 0.5, dtype=cur.dtype, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in scalar_attrs_neg1:
                cur = getattr(self, attr)
                add = torch.full((n_grow,), -1.0, dtype=cur.dtype, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in scalar_attrs_0_1:
                cur = getattr(self, attr)
                add = torch.full((n_grow,), 0.1, dtype=cur.dtype, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in bool_attrs:
                if not hasattr(self, attr):
                    continue
                cur = getattr(self, attr)
                add = torch.zeros(n_grow, dtype=torch.bool, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in long_attrs_neg1:
                cur = getattr(self, attr)
                add = torch.full((n_grow,), -1, dtype=torch.long, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr in long_attrs_classical:
                cur = getattr(self, attr)
                add = torch.full((n_grow,), S_CLASSICAL, dtype=torch.long, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            for attr, spec in random_attrs.items():
                cur = getattr(self, attr)
                dist, scale, offset = spec
                if dist == 'randn':
                    add = torch.randn(n_grow, device=dev) * scale + offset
                else:
                    add = torch.rand(n_grow, device=dev) * scale + offset
                setattr(self, attr, torch.cat([cur, add], dim=0))

            # ── 2-D per-osc state tensors  (n_max, d)
            for attr in ['s', 'future_prediction', '_prev_prediction']:
                if not hasattr(self, attr):
                    continue
                cur = getattr(self, attr)
                if cur is None:
                    continue
                add = torch.zeros(n_grow, d, device=dev)
                setattr(self, attr, torch.cat([cur, add], dim=0))

            # c: unit-norm random directions
            cur = self.c
            cr = torch.randn(n_grow, d, device=dev)
            cr = cr / cr.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.c = torch.cat([cur, cr], dim=0)

            # phase_vec: random phase angles → (cos, sin)
            ph = torch.rand(n_grow, device=dev) * 2 * math.pi
            new_pv = torch.stack([torch.cos(ph), torch.sin(ph)], dim=1)
            self.phase_vec = torch.cat([self.phase_vec, new_pv], dim=0)

            # ── force_channels (n_max, 4)
            if hasattr(self, 'force_channels') and self.force_channels is not None:
                add = torch.zeros(n_grow, 4, device=dev)
                self.force_channels = torch.cat([self.force_channels, add], dim=0)

            # ── scale_memory (n_max, 4, d), s_branches (n_max, 4, d), branch_weights (n_max, 4)
            if hasattr(self, 'scale_memory') and self.scale_memory is not None:
                add = torch.zeros(n_grow, 4, d, device=dev)
                self.scale_memory = torch.cat([self.scale_memory, add], dim=0)
            if hasattr(self, 's_branches') and self.s_branches is not None:
                add = torch.randn(n_grow, 4, d, device=dev) * 0.01
                self.s_branches = torch.cat([self.s_branches, add], dim=0)
            if hasattr(self, 'branch_weights') and self.branch_weights is not None:
                add = torch.ones(n_grow, 4, device=dev) * 0.25
                self.branch_weights = torch.cat([self.branch_weights, add], dim=0)

            # ── _E_history (n_max, 8)
            if hasattr(self, '_E_history') and self._E_history is not None:
                add = torch.zeros(n_grow, 8, device=dev)
                self._E_history = torch.cat([self._E_history, add], dim=0)

            # ── _prev_s, _prev_mag — may be None
            if hasattr(self, '_prev_s') and self._prev_s is not None:
                add = torch.zeros(n_grow, d, device=dev)
                self._prev_s = torch.cat([self._prev_s, add], dim=0)
            if hasattr(self, '_prev_mag') and self._prev_mag is not None:
                add = torch.zeros(n_grow, device=dev)
                self._prev_mag = torch.cat([self._prev_mag, add], dim=0)

            # ── BOND TENSORS (the big ones)
            # C is (old_n, old_n, d, d). Need (new_n, new_n, d, d).
            # Copy existing bonds into top-left corner, new slots get zero bonds.
            new_C = torch.zeros(new_n, new_n, d, d, device=dev)
            new_C[:old_n, :old_n] = self.C
            self.C = new_C

            new_Cmask = torch.zeros(new_n, new_n, dtype=torch.bool, device=dev)
            new_Cmask[:old_n, :old_n] = self.C_mask
            self.C_mask = new_Cmask

            # _teach_bond_sum is (n_classes, n_max, d, d) — only n_max axis grows
            if hasattr(self, '_teach_bond_sum') and self._teach_bond_sum is not None:
                n_cls = self._teach_bond_sum.shape[0]
                add = torch.zeros(n_cls, n_grow, d, d, device=dev)
                self._teach_bond_sum = torch.cat([self._teach_bond_sum, add], dim=1)

            # ── Update n_max — this is the contract that everything else
            # relies on (consolidate, phantom_birth, _learn_bonds, etc.)
            self.n_max = new_n

            return new_n


# Attach learning methods
for _m in ('_valence_per_osc', '_alpha_of_current_state', '_learn_bonds',
           '_prune_bonds', '_identity_drift', '_competitive_c_learning',
           '_contradiction', '_phantom_birth', '_age_and_cull',
           'consolidate_class', 'grow_capacity'):
    setattr(UERFField, _m, getattr(UERFLearning, _m))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — EXPERIENCE LOOP (train + predict)
# ═══════════════════════════════════════════════════════════════════════════════
class UERFExperience:
    """Mixin: experience (training), rehearsal, and prediction."""

    def experience(self, x, teaching_vector=None, n_relax=6, learn=True,
                   bond_chunk=256):
        """
        Present one input pattern to the brain.
        x: (n_sensory,) normalized input vector.
        teaching_vector: (n_classes,) one-hot target (or None for inference).
        n_relax: number of dynamics ticks per experience.
        learn: whether to update bonds.
        """
        with torch.no_grad():
            self.experience_count += 1
            n, d = self.n_max, self.d

            # Device safety: ensure input is on same device as model
            x = x.to(self.device)
            if teaching_vector is not None:
                teaching_vector = teaching_vector.to(self.device)

            x = x / x.norm().clamp(min=1e-6)

            # This creates SPARSE, PATTERN-SPECIFIC representations:
            # - Each input activates a DIFFERENT subset of interior oscillators
            # - Oscillators whose identity (c) aligns with the input get boosted
            # - Oscillators misaligned with input get suppressed more
            # This is the key to preventing catastrophic forgetting:
            # A-pattern oscillators form bonds to teach[0],
            # B-pattern oscillators form bonds to teach[1],
            # and they don't interfere because they're different oscillators.
            interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
            if interior.any():
                cr = self.crystallized()

                # Input-dependent gating: how much does this input
                # "resonate" with each interior oscillator's identity?
                # Use the sensory input projected into d-space
                if self.input_dim is not None:
                    # Compute input signature in d-space
                    # (weighted sum of sensory c vectors)
                    input_sig = (x.unsqueeze(-1) * self.c[:self.input_dim]).sum(0)  # (d,)
                    input_sig = input_sig / input_sig.norm().clamp(min=1e-6)
                    # Resonance: how aligned is each interior osc's c with input?
                    resonance = (self.c * input_sig.unsqueeze(0)).sum(-1)  # (n,)
                    # Soft gate: resonant oscillators decay less, non-resonant decay more
                    # resonance ∈ [-1, 1]; map to gate ∈ [0.3, 1.0]
                    gate = 0.3 + 0.7 * (resonance.clamp(-1, 1) + 1) / 2  # ∈ [0.3, 1.0]
                else:
                    gate = torch.ones(self.n_max, device=self.device)

                # Eq 5 (Master Continuous) dissipation — STATE-SPECIFIC: each
                # regime uses its own loss law (Holographic exp(−S_ent)≈0,
                # Harmonic Q-factor, Thermal exp(E_a/kT), Phantom |1+w|, …).
                # Build current routing ctx so the per-state dissipation drivers
                # (holo fidelity, scale coherence, proper-time) are fresh.
                _decay_ctx = self._build_routing_ctx()
                decay_factor = self.eq5_decay_factor(ctx=_decay_ctx)
                # Apply input-dependent resonance gate (non-resonant decay faster)
                decay_factor = decay_factor * gate
                decay_factor = decay_factor.clamp(0.1, 0.99)
                # UERF PRESERVATION as the FINAL word — crystallized and locked
                # oscillators are the α≈0.998 information-preservation regime and
                # must NOT decay. (Order matters: gate runs before these so it
                # cannot undo them — the bug that made consolidated memory decay.)
                decay_factor = torch.where(cr,
                                           torch.ones_like(decay_factor) * 0.98,
                                           decay_factor)
                decay_factor = torch.where(self._class_locked,
                                           torch.ones_like(decay_factor),
                                           decay_factor)

                self.s[interior] *= decay_factor[interior].unsqueeze(-1)

            # Inject sensory input
            if self.input_dim is not None:
                self.s[:self.input_dim] = x.unsqueeze(-1) * self.c[:self.input_dim]

            # Pin teaching signal during learning
            pin_teach = None
            if teaching_vector is not None and self.t_start is not None:
                pin_teach = teaching_vector.unsqueeze(-1) * self.c[self.t_start:self.t_end]

            # Build context once per experience
            ctx = self._build_routing_ctx()

            # Relax dynamics
            pin_s = self.s[:self.input_dim].clone() if self.input_dim else None
            for tick in range(n_relax):
                self._dynamics_step(ctx, pin_sensory=pin_s,
                                    pin_teaching=pin_teach,
                                    bond_chunk=bond_chunk)
                # Refresh context every other tick for responsiveness
                if tick % 2 == 1:
                    ctx = self._build_routing_ctx()

            # Learning phase
            if learn:
                self._learn_bonds()
                self._identity_drift()

                # Competitive c_i learning: winners specialize to current input.
                # resonance gate AND bond structure, so fast drift misaligns
                # bonds learned earlier in training.
                if self.input_dim is not None:
                    input_sig = (x.unsqueeze(-1) * self.c[:self.input_dim]).sum(0)
                    _lrc = 0.001 if self.fixes.get('fix4') else 0.005
                    self._competitive_c_learning(input_sig=input_sig, lr_c=_lrc)

                # Phantom birth and emergent growth triggers, calibrated to
                # match the contradiction signal's REAL range: with the current
                # formula (max of mean-valence term and variance term), a field
                # that is learning normally sits at ~0.25-0.35; genuine
                # difficulty (V_mean ≤ 0.2 or valence std > 0.2) pushes it past
                # 0.40. The old 0.5 trigger only fired when the field was
                # catastrophically failing, so birth/growth were silently inert
                # through all of Phase 1.
                contradiction = self._contradiction()
                refractory = self.experience_count - self._last_birth_step > 50

                # Phantom birth fires when contradiction > 0.40 (above baseline)
                if contradiction > 0.40 and refractory:
                    n_spawn = max(1, int(contradiction * 4))   # 2-4 spawns

                    # EMERGENT NEUROGENESIS: when the field is full AND
                    # contradiction stays elevated, the tensor substrate
                    # itself signals it needs to expand. The trigger fires
                    # from internal field dynamics — not a human decision.
                    #
                    # Memory preservation: grow_capacity EXTENDS tensors
                    # (concatenates new zeros). Existing oscillator state
                    # and bond patterns are bitwise identical after grow.
                    n_dormant = int((~self.alive_mask).sum().item())
                    grow_refractory = (self.experience_count -
                                       getattr(self, '_last_grow_step', -10000)) > 200
                    # Grow when full AND contradiction is genuinely elevated
                    if n_dormant == 0 and grow_refractory and contradiction > 0.45:
                        # Grow by 20% of current capacity (min 100 slots)
                        n_grow = max(100, int(self.n_max * 0.2))
                        old_n = self.n_max
                        new_n = self.grow_capacity(n_grow=n_grow)
                        self._last_grow_step = self.experience_count
                        # Real osc populate the new slots via _phantom_birth() below
                        # via _phantom_birth(). births counter tracks them.
                    born = self._phantom_birth(n_spawn)
                    if born > 0:
                        self._last_birth_step = self.experience_count

                # Age and cull every 50 steps
                if self.experience_count % 50 == 0:
                    self._age_and_cull()

                # STRATIFIED REPLAY BUFFER (per-class FIFO)
                # Previously FIFO across all classes → buffer became 100%
                # Phase B by step ~750, rehearsal then reinforced ONLY Phase B
                # and erased Phase A. Now: per-class deque of last N samples.
                # Phase B's class buckets fill independently of Phase A's.
                # Rehearsal samples uniformly across populated buckets.
                #
                # If _replay_disabled, skip accumulation entirely — physics
                # must defend memory alone, no replay assistance.
                if (teaching_vector is not None and self.n_classes is not None
                        and not self._replay_disabled):
                    class_id = int(teaching_vector.argmax())
                    if class_id not in self._replay_per_class:
                        self._replay_per_class[class_id] = []
                    self._replay_per_class[class_id].append(
                        (x.clone(), teaching_vector.clone()))
                    if len(self._replay_per_class[class_id]) > self._replay_capacity_per_class:
                        self._replay_per_class[class_id].pop(0)

                # Rehearsal every 10 steps (replays prior class samples)
                # Also skipped when _replay_disabled.
                if not self._replay_disabled:
                    total_replays = sum(len(v) for v in self._replay_per_class.values())
                    if self.experience_count % 10 == 0 and total_replays > 5:
                        self._rehearse()

    def _rehearse(self, n_replay=6):
        """
        Replay stored patterns from the STRATIFIED per-class buffer to
        consolidate memory across all classes seen so far. Each rehearsal
        samples uniformly across classes, ensuring Phase A classes get
        rehearsed during Phase B (the fix for catastrophic forgetting).
        """
        with torch.no_grad():
            if not self._replay_per_class:
                return
            # Flatten all class buffers
            all_examples = []
            for buf in self._replay_per_class.values():
                all_examples.extend(buf)
            if len(all_examples) < 2:
                return

            n = min(n_replay, len(all_examples))
            indices = torch.randperm(len(all_examples))[:n].tolist()
            for idx in indices:
                x_rep, tv_rep = all_examples[idx]

                if self.input_dim is not None:
                    self.s[:self.input_dim] = x_rep.unsqueeze(-1) * self.c[:self.input_dim]

                pin_teach = None
                if tv_rep is not None and self.t_start is not None:
                    pin_teach = tv_rep.unsqueeze(-1) * self.c[self.t_start:self.t_end]

                pin_s = self.s[:self.input_dim].clone() if self.input_dim else None
                ctx = self._build_routing_ctx()

                # Slightly more ticks for replay (consolidation should be deep)
                for _ in range(4):
                    self._dynamics_step(ctx, pin_sensory=pin_s,
                                        pin_teaching=pin_teach)

                # Full learning rate (consolidation reinforces strongly)
                self._learn_bonds(lr=0.08, decay=1e-4)

    def predict(self):
        """Read out the teaching slot magnitudes as class scores."""
        if self.t_start is None:
            return None
        teach_s = self.s[self.t_start:self.t_end]
        return teach_s.norm(dim=-1)

    def eval_predict(self, x, n_relax=None, bond_chunk=256,
                     return_scores=False):
        """
        Non-destructive evaluation. Save ALL state, run inference, restore.
        fix5: default n_relax 10 instead of 6 (more ticks for bonds to drive
        teach slots from zero). Explicit n_relax arg always overrides.
        return_scores: return the full per-class score vector instead of
        the argmax index (None if there are no teach slots).
        """
        if n_relax is None:
            n_relax = 10 if self.fixes.get('fix5') else 6
        # Save complete state
        save = {
            's': self.s.clone(),
            'state_id': self.state_id.clone(),
            'phase_vec': self.phase_vec.clone(),
            'gamma': self.gamma.clone(),
            'proper_time': self.proper_time.clone(),
            'branch_weights': self.branch_weights.clone(),
            'future_prediction': self.future_prediction.clone(),
            '_prev_prediction': self._prev_prediction.clone(),
            'force_channels': self.force_channels.clone(),
            'scale_memory': self.scale_memory.clone(),
            'vacuum_energy': self.vacuum_energy.clone(),
            '_prev_mag': self._prev_mag.clone() if self._prev_mag is not None else None,
            '_prev_s': self._prev_s.clone() if self._prev_s is not None else None,
            'local_T': self.local_T.clone(),
            'local_v2c2': self.local_v2c2.clone(),
            'local_w': self.local_w.clone(),
            '_E_history': self._E_history.clone(),
            's_branches': self.s_branches.clone(),
            'state_weights': self.state_weights.clone(),
            '_input_running_mean': self._input_running_mean.clone() if self._input_running_mean is not None else None,
            '_input_running_var': self._input_running_var.clone(),
            '_last_birth_step': self._last_birth_step,
            '_valence_accumulator': self._valence_accumulator.clone(),
            '_valence_count': self._valence_count.clone(),
        }

        # Device safety: ensure input is on same device as model
        x = x.to(self.device)

        # State-dependent decay. fix1: additionally apply the input-resonance
        # gate that experience() uses (so eval recreates the sparse pattern-
        # specific activation). NOTE: proven inert alone in A/B (the gate is
        # applied once pre-loop and the relaxation loop washes it out), but
        # kept toggleable for the full ablation — it may matter in combination.
        x = x / x.norm().clamp(min=1e-6)
        interior = self.alive_mask & ~self._is_sensory & ~self._is_teaching
        if interior.any():
            cr = self.crystallized()

            if self.fixes.get('fix1') and self.input_dim is not None:
                input_sig = (x.unsqueeze(-1) * self.c[:self.input_dim]).sum(0)
                input_sig = input_sig / input_sig.norm().clamp(min=1e-6)
                resonance = (self.c * input_sig.unsqueeze(0)).sum(-1)
                gate = 0.3 + 0.7 * (resonance.clamp(-1, 1) + 1) / 2
            else:
                gate = torch.ones(self.n_max, device=self.device)

            # Eq 5 dissipation — STATE-SPECIFIC, same as experience() (each
            # regime's own loss law), so inference and training match.
            _decay_ctx = self._build_routing_ctx()
            decay_factor = self.eq5_decay_factor(ctx=_decay_ctx)
            # Gate and clamp FIRST, then lock/crystal preservation as the FINAL
            # word (otherwise the gate multiply would undo the lock and
            # consolidated memory would decay during eval, weakening readout).
            decay_factor = (decay_factor * gate).clamp(0.1, 0.99)
            decay_factor = torch.where(cr, torch.ones_like(decay_factor) * 0.98,
                                       decay_factor)
            decay_factor = torch.where(self._class_locked,
                                       torch.ones_like(decay_factor),
                                       decay_factor)
            self.s[interior] *= decay_factor[interior].unsqueeze(-1)

        # CRITICAL FIX: Reset teaching slots to ZERO before eval.
        # During training, teaching slots are PINNED to the target.
        # During eval, they must be DRIVEN by bonds from interior.
        # If we don't reset them, they retain the last training pin value
        # and the argmax always returns the last-trained class.
        if self.t_start is not None:
            self.s[self.t_start:self.t_end] = 0.0

        if self.input_dim is not None:
            self.s[:self.input_dim] = x.unsqueeze(-1) * self.c[:self.input_dim]

        pin_s = self.s[:self.input_dim].clone() if self.input_dim else None
        ctx = self._build_routing_ctx()

        # More relaxation ticks for eval (bonds need time to drive teaching)
        for tick in range(n_relax):
            self._dynamics_step(ctx, pin_sensory=pin_s, pin_teaching=None,
                                bond_chunk=bond_chunk)
            # Refresh context midway
            if tick == n_relax // 2:
                ctx = self._build_routing_ctx()

        scores = self.predict()
        result = int(scores.argmax()) if scores is not None else -1

        # Restore everything
        for attr, val in save.items():
            if val is not None:
                setattr(self, attr, val)

        if return_scores:
            return scores.clone() if scores is not None else None
        return result

    def predict_scores(self, x, n_relax=None, bond_chunk=256):
        """Non-destructive evaluation returning the full per-class score
        vector (teach-slot magnitudes) rather than the argmax index."""
        return self.eval_predict(x, n_relax=n_relax, bond_chunk=bond_chunk,
                                 return_scores=True)


# Attach experience methods
for _m in ('experience', '_rehearse', 'predict', 'predict_scores', 'eval_predict'):
    setattr(UERFField, _m, getattr(UERFExperience, _m))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — CHECKPOINT / SAVE / RESUME
# ═══════════════════════════════════════════════════════════════════════════════
class UERFCheckpoint:
    """Save/load the entire brain state."""

    SAVE_ATTRS = (
        # config
        'n_max', 'd', 'input_dim', 'n_classes', 't_start', 't_end',
        '_holo_area_limit',
        'experience_count', 'births', 'deaths', 'crystallization_events',
        # identity tensors
        'theta', 'S', 'f', 'N', 'a0', 'age', 'alive_mask',
        # state
        's', 'c', 'phase_vec', 'state_id',
        # bonds
        'C', 'C_mask',
        # slot tags
        '_is_sensory', '_is_teaching',
        # holographic
        '_holo_projector',
        # auxiliary state variables
        'proper_time', 'gamma', 'branch_weights', 'future_prediction',
        '_prev_prediction',
        'force_channels', 'scale_memory', 'vacuum_energy',
        '_prev_mag', '_prev_s',
        # routing cache
        'state_weights',
        # local physics, cross-state coupling, parallel branches
        'local_T', 'local_v2c2', 'local_w', '_E_history',
        'C_states', 's_branches',
        # birth refractory + crystallization
        '_last_birth_step', '_last_grow_step',
        '_valence_accumulator', '_valence_count',
        '_crystallization_threshold',
        '_teach_bond_count',
    )

    def save_checkpoint(self, path):
        """Save full brain state to a .pt file."""
        ckpt = {}
        for attr in UERFCheckpoint.SAVE_ATTRS:
            v = getattr(self, attr, None)
            if v is None:
                ckpt[attr] = None
            elif isinstance(v, torch.Tensor):
                ckpt[attr] = v.detach().cpu().clone()
            else:
                ckpt[attr] = v
        # Save stratified replay buffer
        replay_data = {}
        for cls_id, buf in self._replay_per_class.items():
            replay_data[cls_id] = [(x.cpu(), t.cpu()) for x, t in buf]
        ckpt['_replay_per_class'] = replay_data
        # Save teach bond accumulators
        if self._teach_bond_sum is not None:
            ckpt['_teach_bond_sum'] = self._teach_bond_sum.detach().cpu().clone()
        ckpt['_teach_bond_count'] = self._teach_bond_count.detach().cpu().clone()
        # Save class lock state
        ckpt['_class_locked'] = self._class_locked.detach().cpu().clone()
        ckpt['_locked_class'] = self._locked_class.detach().cpu().clone()
        # Save ablation fix toggles so a checkpoint remembers its config
        ckpt['fixes'] = dict(self.fixes)
        torch.save(ckpt, path)
        return path

    @staticmethod
    def load_checkpoint(path, device=None):
        """Reconstruct a UERFField from a checkpoint."""
        dev = device or DEVICE
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        f = UERFField(
            n_max=ckpt['n_max'],
            n_initial=1,
            d=ckpt['d'],
            input_dim=ckpt['input_dim'],
            n_classes=ckpt['n_classes'],
            device=dev,
        )
        for attr in UERFCheckpoint.SAVE_ATTRS:
            v = ckpt.get(attr, None)
            if isinstance(v, torch.Tensor):
                setattr(f, attr, v.to(dev))
            elif v is not None:
                setattr(f, attr, v)
        # Restore stratified replay buffer
        if '_replay_per_class' in ckpt and ckpt['_replay_per_class']:
            f._replay_per_class = {}
            for cls_id, buf in ckpt['_replay_per_class'].items():
                f._replay_per_class[cls_id] = [(x.to(dev), t.to(dev)) for x, t in buf]
        elif '_replay_inputs' in ckpt and ckpt['_replay_inputs']:
            # Backwards compat: old FIFO checkpoint — distribute into classes
            for x, t in ckpt['_replay_inputs']:
                cls_id = int(t.argmax())
                if cls_id not in f._replay_per_class:
                    f._replay_per_class[cls_id] = []
                f._replay_per_class[cls_id].append((x.to(dev), t.to(dev)))
        # Restore teach bond accumulators
        if '_teach_bond_sum' in ckpt and ckpt['_teach_bond_sum'] is not None:
            f._teach_bond_sum = ckpt['_teach_bond_sum'].to(dev)
        if '_teach_bond_count' in ckpt:
            f._teach_bond_count = ckpt['_teach_bond_count'].to(dev)
        # Restore class locks
        if '_class_locked' in ckpt:
            f._class_locked = ckpt['_class_locked'].to(dev)
        if '_locked_class' in ckpt:
            f._locked_class = ckpt['_locked_class'].to(dev)
        # Restore ablation fix toggles (default all-False for old checkpoints)
        if 'fixes' in ckpt and isinstance(ckpt['fixes'], dict):
            f.fixes = dict(ckpt['fixes'])
        return f


# Attach checkpoint methods
for _m in ('save_checkpoint',):
    setattr(UERFField, _m, getattr(UERFCheckpoint, _m))
UERFField.load_checkpoint = staticmethod(UERFCheckpoint.load_checkpoint)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    'UERFField', 'SensoryProjection', 'DEVICE', 'STATE_NAMES',
    'S_CLASSICAL', 'S_TOROIDAL', 'S_HARMONIC', 'S_HOLOGRAPHIC', 'S_PHANTOM',
    'S_THERMAL', 'S_RELATIVISTIC', 'S_VACUUM', 'S_FRACTAL', 'S_MULTIVERSAL',
    'S_RETROCAUSAL', 'S_HOLOADS', 'S_CUBIT', 'S_ELEMENTAL', 'S_QUANTUM',
    'S_QUBIT', 'S_TEMPORAL', 'S_CONSCIOUSNESS',
    'ALPHA_RANGES', 'alpha_for_state', 'alpha_core', 'valence',
]
