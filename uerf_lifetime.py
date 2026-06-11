"""
UERF — Cross-Domain Lifetime Training
═════════════════════════════════════
Trains a single brain through 4 phases to test catastrophic forgetting
across radically different input distributions:

  Phase 1: MNIST  (10 classes)       — teach slots 0-9
  Phase 2: CIFAR-100 (100 classes)   — teach slots 10-109
  Phase 3: Tiny ImageNet (200 cls)   — teach slots 110-309
  Phase 4: MNIST RE-EVAL (no train)  — does the brain still remember Phase 1?

If Phase 4 MNIST accuracy is close to Phase 1's, the framework genuinely
preserves memory across domain shifts. This is the cross-domain catastrophic
forgetting test.

Universal encoder
─────────────────
All three datasets get encoded through a frozen ImageNet-pretrained ResNet-18
(stripped of its classifier head, output = 512-dim feature vector). This puts
MNIST, CIFAR, and TinyImageNet into the SAME semantic feature space so the
brain doesn't have to relearn pixel statistics from scratch each phase.

  MNIST (1×28×28)   → upscale to 3×224×224 → ResNet → 512-d features
  CIFAR (3×32×32)   → upscale to 3×224×224 → ResNet → 512-d features
  TI (3×64×64)      → upscale to 3×224×224 → ResNet → 512-d features
  → SensoryProjection (512 → n_sensory=128)
  → UERFField

Pre-allocation
──────────────
The brain is built with n_classes=310 from the start. Phase 1 uses only
teach slots 0-9 (rest dormant but reserved). Phase 2 activates 10-109.
Phase 3 activates 110-309. Phase 4 is read-only evaluation.

This avoids the slot-reindexing problem entirely. Locked Phase 1 oscillators
stay locked. The C tensor never needs surgery.

USAGE
─────
  modal run modal_lifetime.py --phase 1   # MNIST only (~30 min A100)
  modal run modal_lifetime.py --phase 2   # CIFAR-100 (~2-3 hours)
  modal run modal_lifetime.py --phase 3   # Tiny ImageNet (~3-4 hours)
  modal run modal_lifetime.py --phase 4   # re-eval MNIST (~1 min)

Each phase resumes from lifetime_main.pt (saved by previous phase).
"""

import os, sys, time, json, argparse
import torch
import numpy as np

# ── Sentry ────────────────────────────────────────────────────────────────────
try:
    import sentry_sdk
    _SENTRY_DSN = os.environ.get(
        "SENTRY_DSN_OVERRIDE",
        "https://54476bc9ee945366dbde8c51709ac74b@o4511431662829568.ingest.us.sentry.io/4511448029134848"
    )
    sentry_sdk.init(dsn=_SENTRY_DSN, traces_sample_rate=0.1,
                    release="uerf-lifetime-v1")
    SENTRY_OK = True
    print("[sentry] telemetry enabled")
except ImportError:
    SENTRY_OK = False

def sentry_log(category, msg, **data):
    print(f"[{category}] {msg}")
    if SENTRY_OK:
        try:
            sentry_sdk.add_breadcrumb(category=category, message=msg, level="info", data=data)
        except Exception:
            pass

def sentry_event(name, **data):
    if SENTRY_OK:
        try:
            # Use isolation_scope (modern API) — push_scope causes context
            # bleed and VRAM leaks in long-running loops (40K+ steps).
            with sentry_sdk.isolation_scope() as scope:
                for k, v in data.items():
                    scope.set_extra(k, v)
                sentry_sdk.capture_message(name, level="info")
        except Exception:
            pass

# ── Paths ─────────────────────────────────────────────────────────────────────
if os.path.isdir('/uerf-output'):
    BASE_DIR = '/uerf-output'
elif os.path.isdir('/kaggle/working'):
    BASE_DIR = '/kaggle/working'
else:
    BASE_DIR = '.'

CKPT_DIR = os.path.join(BASE_DIR, 'lifetime_checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'lifetime_data')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CKPT_LIFETIME = os.path.join(CKPT_DIR, 'lifetime_main.pt')
LOG_PATH = os.path.join(CKPT_DIR, 'lifetime_log.json')

# ── Class index ranges ────────────────────────────────────────────────────────
MNIST_RANGE  = (0, 10)      # 10 classes
CIFAR_RANGE  = (10, 110)    # 100 classes
TI_RANGE     = (110, 310)   # 200 classes (Tiny ImageNet has 200)
N_CLASSES    = 310          # total reserved

# ── Brain config ──────────────────────────────────────────────────────────────
CONFIG = {
    # ── BRAIN DIMENSIONS ──────────────────────────────────────────────────
    # n_max = HARDWARE CEILING (the absolute physical tensor size)
    # n_initial = INITIAL ALIVE COUNT (the seed; dormant slots are real
    #             room for grow_capacity/_phantom_birth to populate)
    'n_sensory'      : 128,           # bond input width
    'd'              : 48,            # per-osc state dim
    'n_max'          : 1800,          # PHYSICAL CEILING — set to A100-80GB limit
    'n_initial'      : 500,           # 128 sensors + 310 teach + 62 plastic interior to start
                                      # 1300 dormant slots available for growth
    'n_classes'      : N_CLASSES,     # 310 teach slots reserved for entire lifetime

    'n_relax'        : 6,
    'n_lock_per_cls' : 4,             # 4 osc × 310 classes = 1240 lock budget MAX
                                      # leaves ~560 plastic for ongoing learning

    # ── REPLAY BUFFER POLICY ──────────────────────────────────────────────
    # Flaw A (per audit): if patent claim is that physics SOLVE catastrophic
    # forgetting, then replay buffer is competing evidence. Disable it for
    # the gauntlet to prove the architecture alone preserves memory.
    'disable_replay' : True,          # set True for clean physics-only test

    # ── PER-PHASE STEPS ───────────────────────────────────────────────────
    'phase1_mnist_steps'    : 10000,
    'phase2_cifar_steps'    : 30000,
    'phase3_ti_steps'       : 40000,
    'eval_samples_per_class': 20,

    'log_every'      : 500,
    'ckpt_every'     : 2000,
}

# Nominal cumulative experience-count boundaries. Used ONLY as a fallback for
# resume anchoring when lifetime_log.json has no record of the previous phase.
# The real anchor is the experience_count logged by 'phase_{N-1}_complete' —
# phases can legitimately overshoot their budget (e.g. a re-run with an older
# script), and anchoring on the log self-heals that drift instead of silently
# skipping steps in the next phase.
_PHASE_START = {
    1: 0,
    2: CONFIG['phase1_mnist_steps'],
    3: CONFIG['phase1_mnist_steps'] + CONFIG['phase2_cifar_steps'],
}


def phase_anchor(phase_n):
    """
    Return the experience_count at which phase_n's training window begins.

    Phase 1 always anchors at 0 (fresh brain). For later phases, read the
    last 'phase_{N-1}_complete' event from lifetime_log.json and use its
    recorded experience_count — the true count when the previous phase
    finished, regardless of overshoot. Falls back to the nominal
    _PHASE_START boundary (with a printed warning) if no event exists.
    """
    if phase_n == 1:
        return 0
    prev_event = f'phase_{phase_n - 1}_complete'
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                log = json.load(f)
            for entry in reversed(log):
                if entry.get('event') == prev_event and 'experience_count' in entry:
                    anchor = int(entry['experience_count'])
                    print(f"[anchor] phase {phase_n} anchored at experience_count="
                          f"{anchor} (from '{prev_event}' log event)")
                    return anchor
        except Exception as e:
            print(f"[anchor] WARNING: failed to read {LOG_PATH}: {e}")
    nominal = _PHASE_START[phase_n]
    print(f"[anchor] WARNING: no '{prev_event}' event in log; "
          f"falling back to nominal boundary {nominal}")
    return nominal


def _phase_already_complete(phase_n):
    """Return True if phase_{phase_n}_complete event exists in lifetime_log.json."""
    event = f'phase_{phase_n}_complete'
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                log = json.load(f)
            if any(e.get('event') == event for e in log):
                print(f"[guard] phase {phase_n} already complete in log — "
                      f"skipping consolidation and log_event")
                return True
        except Exception:
            pass
    return False

# Memory budget at this config:
#   C tensor = n_max² × d² × 4 bytes = 1800² × 48² × 4 = ~30 GB
# Requires A100-80GB. If you only have A100-40GB:
#   set n_max=1200, d=40 → C = 1200² × 40² × 4 = ~9.2 GB

# ── Universal encoder (frozen ResNet-18) ──────────────────────────────────────

class UniversalEncoder:
    """
    Frozen ImageNet-pretrained ResNet-18 used as a feature extractor.
    Maps any image (any size, 1 or 3 channels) to a fixed 512-d feature
    vector. Output is normalized to unit length.

    Same model handles MNIST, CIFAR-100, Tiny ImageNet — so the brain's
    sensor projection sees comparable feature statistics across domains.
    """
    def __init__(self, device=None):
        import torchvision
        import torchvision.transforms as T
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained ResNet-18, strip the classifier
        net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        # Replace classifier with identity → output is 512-d penultimate features
        net.fc = torch.nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net.to(self.device)

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(self, img_batch):
        """
        img_batch: (B, C, H, W) tensor in [0, 1] range.
          C can be 1 (MNIST) or 3 (CIFAR / TI). Will be expanded to 3.
          H, W can be anything — will be resized to 224×224.
        Returns: (B, 512) unit-norm feature vectors.
        """
        x = img_batch.to(self.device)
        # Expand grayscale to RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Resize to 224×224
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224),
                                                 mode='bilinear', align_corners=False)
        # Normalize
        x = (x - self.mean) / self.std
        # Forward pass
        feat = self.net(x)                              # (B, 512)
        # Unit normalize
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat

# ── Data loaders ──────────────────────────────────────────────────────────────

# ── Data loaders (cached at module level — Flaw G fix) ───────────────────────
_DATASET_CACHE = {}

def load_mnist():
    if 'mnist' in _DATASET_CACHE:
        return _DATASET_CACHE['mnist']
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor()])
    tr = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
    te = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    _DATASET_CACHE['mnist'] = (tr, te)
    return tr, te

def load_cifar100():
    if 'cifar100' in _DATASET_CACHE:
        return _DATASET_CACHE['cifar100']
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor()])
    tr = torchvision.datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf)
    _DATASET_CACHE['cifar100'] = (tr, te)
    return tr, te

def load_tiny_imagenet():
    """
    Tiny ImageNet: 200 classes, 3×64×64 images.
      train/  : 500 images per class in ImageFolder structure
      val/    : 10000 images (50/class), labels in val_annotations.txt

    Returns (train_dataset, val_dataset) — properly separated.

    NOTE ON DATA LEAK CAVEAT (audit Flaw B):
    TinyImageNet classes are a subset of ImageNet-1K, so the frozen
    ResNet-18 used as encoder has SEEN these classes during pretraining.
    This inflates absolute accuracy numbers BUT does NOT invalidate the
    CONTINUAL LEARNING test: we're measuring whether MNIST and CIFAR
    knowledge survives Phase 3 training, not whether TI absolute accuracy
    is meaningful. The retention metric remains valid even if TI itself
    benefits from feature leakage.
    """
    if 'tiny_imagenet' in _DATASET_CACHE:
        return _DATASET_CACHE['tiny_imagenet']
    import urllib.request, zipfile
    ti_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200')
    if not os.path.isdir(ti_dir):
        print('[data] downloading Tiny ImageNet (~250 MB)...')
        zip_path = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')
        urllib.request.urlretrieve(
            'http://cs231n.stanford.edu/tiny-imagenet-200.zip', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)

    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor()])

    # Training set (ImageFolder structure)
    tr = torchvision.datasets.ImageFolder(os.path.join(ti_dir, 'train'), transform=tf)

    # Validation set — parse val_annotations.txt to assign labels
    # Format: filename<TAB>class_id<TAB>x1<TAB>y1<TAB>x2<TAB>y2
    val_ann_path = os.path.join(ti_dir, 'val', 'val_annotations.txt')
    val_img_dir = os.path.join(ti_dir, 'val', 'images')

    # Build class_name → class_idx mapping (same as train)
    class_to_idx = tr.class_to_idx

    class TinyImageNetVal(torch.utils.data.Dataset):
        def __init__(self, ann_path, img_dir, class_to_idx, transform):
            self.samples = []
            self.transform = transform
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    fname = parts[0]
                    class_name = parts[1]
                    if class_name in class_to_idx:
                        path = os.path.join(img_dir, fname)
                        label = class_to_idx[class_name]
                        self.samples.append((path, label))
            self.targets = [s[1] for s in self.samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            from PIL import Image
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    te = TinyImageNetVal(val_ann_path, val_img_dir, class_to_idx, tf)
    print(f"[data] TinyImageNet: train={len(tr)}, val={len(te)}, classes={len(class_to_idx)}")

    _DATASET_CACHE['tiny_imagenet'] = (tr, te)
    return tr, te

# ── Brain helpers ─────────────────────────────────────────────────────────────

def set_active_teach_slots(field, n_active_classes):
    """
    Control which teaching slots are ALIVE for the current phase.

    The brain is built with all CONFIG['n_classes'] (310) teach POSITIONS
    allocated so the tensors are the right size for the whole lifetime and
    checkpoints stay compatible. But a teach slot for a class the brain has
    NOT encountered yet should be DORMANT, not alive — an alive slot is
    routed, evolved, and learned-over every tick, so carrying 300 unborn
    classes through Phase 1 (which only uses 10) is pure wasted compute and
    is exactly why Phase 1 ran at ~1.1 ex/s instead of ~4+.

    This activates teach slots [0 : n_active_classes] and forces the rest
    dormant. It only touches teaching slots (never sensory/interior) and uses
    the brain's existing alive_mask — no change to the brain itself. Idempotent
    and monotonic across phases (Phase 2 activates more, never deactivates).
    """
    if field.t_start is None:
        return
    t0 = field.t_start
    n_total_teach = field.t_end - field.t_start
    n_active = min(n_active_classes, n_total_teach)

    # Active phase teach slots: alive. Future (not-yet-seen) teach slots: dormant.
    active_idx = torch.arange(t0, t0 + n_active, device=field.device)
    future_idx = torch.arange(t0 + n_active, field.t_end, device=field.device)

    field.alive_mask[active_idx] = True
    # Only dormantize FUTURE slots that aren't locked (locked = already
    # consolidated from a prior phase; must never be touched).
    if len(future_idx) > 0:
        not_locked = ~field._class_locked[future_idx]
        to_dormant = future_idx[not_locked]
        field.alive_mask[to_dormant] = False

    n_alive_total = int(field.alive_mask.sum())
    print(f"[teach-slots] activated classes 0-{n_active-1} "
          f"({n_active} alive teach, {len(future_idx)} dormant reserved); "
          f"total alive nodes now {n_alive_total}")


def _configure_field(field):
    """Apply all ablation fixes and rebuild C_states to match the doc's coupling spec.
    Called after every brain build or load so the config is always current.
    """
    field.fixes = {k: True for k in field.fixes}
    print(f"[config] field.fixes = {field.fixes}")


def build_fresh_brain():
    """Build the lifetime brain pre-allocated for all phases."""
    sys.path.insert(0, BASE_DIR)
    sys.path.insert(0, '.')
    from uerf_brain import UERFField
    field = UERFField(
        n_max       = CONFIG['n_max'],
        n_initial   = CONFIG['n_initial'],
        input_dim   = CONFIG['n_sensory'],
        n_classes   = CONFIG['n_classes'],
        d           = CONFIG['d'],
    )
    # Apply replay policy from CONFIG
    field._replay_disabled = CONFIG.get('disable_replay', False)
    _configure_field(field)
    print(f"[brain] built fresh: n_max={CONFIG['n_max']}, d={CONFIG['d']}, "
          f"n_classes={CONFIG['n_classes']}, "
          f"teach slots {field.t_start}-{field.t_end}, "
          f"replay_disabled={field._replay_disabled}")
    return field

def load_or_build_brain():
    sys.path.insert(0, BASE_DIR)
    sys.path.insert(0, '.')
    from uerf_brain import UERFField
    if os.path.exists(CKPT_LIFETIME):
        print(f"[brain] resuming from {CKPT_LIFETIME}")
        field = UERFField.load_checkpoint(CKPT_LIFETIME)
        # Verify dimensions match config
        if field.n_max != CONFIG['n_max'] or field.d != CONFIG['d']:
            raise ValueError(
                f"Checkpoint dims (n_max={field.n_max}, d={field.d}) don't match "
                f"CONFIG (n_max={CONFIG['n_max']}, d={CONFIG['d']}). "
                "Delete the checkpoint or fix CONFIG to match.")
        # Re-apply replay policy and force all ablation fixes on (in case CONFIG
        # or brain code changed since the checkpoint was written).
        field._replay_disabled = CONFIG.get('disable_replay', False)
        _configure_field(field)
        print(f"[brain] loaded: experience={field.experience_count}, "
              f"alive={int(field.alive_mask.sum())}, "
              f"locked={int(field._class_locked.sum())}, "
              f"bonds={int(field.C_mask.sum())}, "
              f"replay_disabled={field._replay_disabled}")
        return field
    else:
        return build_fresh_brain()

# ── Universal training loop ───────────────────────────────────────────────────

def train_phase(field, encoder, dataset, class_offset, n_classes_in_phase,
                n_steps, label, step_offset=0):
    """
    Train the brain on a dataset for n_steps. The teach vector for an image
    of dataset-class c gets pinned to brain-class (c + class_offset).

    encoder:        UniversalEncoder
    dataset:        torchvision Dataset with (img_tensor, label) items
    class_offset:   index where this dataset's classes start in the brain
    n_classes_in_phase: how many classes this dataset has (e.g. 100 for CIFAR)
    n_steps:        number of training examples to feed
    label:          string like "Phase 2 CIFAR-100" for logs
    step_offset:    absolute step number at the start of this call (for resume)
    """
    from uerf_brain import SensoryProjection

    print(f"\n[{label}] training {n_steps} steps "
          f"(offset={step_offset}), brain classes "
          f"{class_offset}-{class_offset + n_classes_in_phase - 1}")

    # Build (or reuse) the projection from feature space → sensors.
    # CRITICAL: the same projection must be used across phases (so sensor
    # alignment is preserved). We seed it deterministically.
    proj = SensoryProjection(512, CONFIG['n_sensory'], seed=42)
    proj.to(field.device)

    # Build a sample pool from the dataset
    print(f"[{label}] indexing dataset...")
    # For ImageFolder (Tiny ImageNet) targets are stored differently
    if hasattr(dataset, 'targets'):
        all_labels = dataset.targets
        if torch.is_tensor(all_labels):
            all_labels = all_labels.tolist()
    elif hasattr(dataset, '_labels'):
        all_labels = dataset._labels
    else:
        all_labels = [dataset[i][1] for i in range(len(dataset))]

    pool_idx = list(range(len(all_labels)))
    print(f"[{label}] pool size: {len(pool_idx)}")

    t0 = time.time()
    bsz = 8  # batch encoding for speed
    teach = torch.zeros(CONFIG['n_classes'], device=field.device)

    for step in range(0, n_steps, bsz):
        # Pick batch
        batch_pos = torch.randint(0, len(pool_idx), (bsz,)).tolist()
        batch_imgs = []
        batch_lbls = []
        for p in batch_pos:
            img, lbl = dataset[pool_idx[p]]
            batch_imgs.append(img)
            batch_lbls.append(lbl)
        batch = torch.stack(batch_imgs)              # (bsz, C, H, W)

        # Encode through ResNet
        feats = encoder.encode(batch)                # (bsz, 512)

        # Project to sensors and feed one at a time
        for i in range(bsz):
            if step + i >= n_steps:
                break
            x = proj(feats[i])
            global_cls = batch_lbls[i] + class_offset
            teach.zero_()
            teach[global_cls] = 1.0
            field.experience(x, teach, n_relax=CONFIG['n_relax'])

        actual_step = step_offset + step + bsz
        if actual_step % CONFIG['log_every'] < bsz:
            elapsed = time.time() - t0
            rate = (step + bsz) / elapsed
            r = field.report()
            top = max(r['state_dist'].items(), key=lambda kv: kv[1])[0] if r['state_dist'] else '?'
            print(f"  step {actual_step:>6}  {rate:>5.1f} ex/s  "
                  f"bonds={r['bonds']:>5}  alive={r['alive']:>4}  "
                  f"top_state={top}  locked={int(field._class_locked.sum())}")
            sentry_log(f"{label.lower().replace(' ', '_')}",
                       f"step {actual_step}",
                       step=actual_step, bonds=r['bonds'], alive=r['alive'],
                       top_state=top, state_dist=str(r['state_dist'])[:200],
                       locked=int(field._class_locked.sum().item()))

        if actual_step % CONFIG['ckpt_every'] < bsz:
            field.save_checkpoint(CKPT_LIFETIME)
            print(f"  [ckpt] saved at step {actual_step}")

    field.save_checkpoint(CKPT_LIFETIME)
    print(f"[{label}] complete. Saved {CKPT_LIFETIME}")

def eval_phase(field, encoder, dataset, class_offset, n_classes_in_phase,
               label, n_per_class=None):
    """Evaluate accuracy on a dataset given the brain's current state."""
    if n_per_class is None:
        n_per_class = CONFIG['eval_samples_per_class']
    from uerf_brain import SensoryProjection
    proj = SensoryProjection(512, CONFIG['n_sensory'], seed=42)
    proj.to(field.device)

    if hasattr(dataset, 'targets'):
        all_labels = dataset.targets
        if torch.is_tensor(all_labels):
            all_labels = all_labels.tolist()
    elif hasattr(dataset, '_labels'):
        all_labels = dataset._labels
    else:
        all_labels = [dataset[i][1] for i in range(len(dataset))]

    # Sample n_per_class examples per class
    from collections import defaultdict
    by_class = defaultdict(list)
    for i, lbl in enumerate(all_labels):
        if len(by_class[lbl]) < n_per_class:
            by_class[lbl].append(i)
        if all(len(v) >= n_per_class for v in by_class.values()) and \
           len(by_class) >= n_classes_in_phase:
            break

    print(f"\n[eval {label}] evaluating {sum(len(v) for v in by_class.values())} examples "
          f"across {len(by_class)} classes...")

    total = 0
    correct = 0
    from collections import Counter
    pred_dist = Counter()
    correct_per_class = Counter()
    total_per_class = Counter()

    t0 = time.time()
    for cls, idxs in by_class.items():
        for idx in idxs:
            img, lbl = dataset[idx]
            feat = encoder.encode(img.unsqueeze(0))[0]   # (512,)
            x = proj(feat)
            pred_global = field.eval_predict(x)
            # Brain returns global class index; convert back to dataset class
            pred_in_phase = pred_global - class_offset
            target = lbl
            total += 1
            total_per_class[cls] += 1
            if pred_in_phase == target:
                correct += 1
                correct_per_class[cls] += 1
            pred_dist[pred_global] += 1
    acc = 100.0 * correct / max(total, 1)
    print(f"[eval {label}] {correct}/{total} = {acc:.2f}% in {time.time()-t0:.0f}s")
    print(f"[eval {label}] prediction distribution top 10: "
          f"{dict(pred_dist.most_common(10))}")
    return acc

def consolidate(field, class_offset, n_classes_in_phase, label):
    """Consolidate all classes in a phase range."""
    print(f"\n[consolidate {label}] locking representations...")
    n_locked = 0
    for c in range(class_offset, class_offset + n_classes_in_phase):
        n = field.consolidate_class(c, n_lock=CONFIG['n_lock_per_cls'])
        n_locked += n
    print(f"[consolidate {label}] locked {n_locked} oscillators")
    sentry_log("consolidate", f"{label}: {n_locked} locked",
               total_locked=int(field._class_locked.sum().item()))

# ── Phase runners ─────────────────────────────────────────────────────────────

def run_phase_1():
    """MNIST baseline. Trains 10K steps on MNIST 0-9, consolidates, evaluates."""
    field = load_or_build_brain()
    encoder = UniversalEncoder()
    tr, te = load_mnist()

    # Only MNIST's 10 teach slots are alive in Phase 1; the other 300 stay
    # dormant (reserved positions) so we don't pay to evolve unborn classes.
    set_active_teach_slots(field, MNIST_RANGE[0] + 10)

    p1_anchor = phase_anchor(1)
    p1_budget = CONFIG['phase1_mnist_steps']
    p1_done   = max(0, field.experience_count - p1_anchor)
    p1_remain = max(0, p1_budget - p1_done)
    if p1_remain > 0:
        print(f"[Phase 1] resuming from step {p1_done}/{p1_budget} — {p1_remain} steps remaining")
        train_phase(field, encoder, tr,
                    class_offset=MNIST_RANGE[0],
                    n_classes_in_phase=10,
                    n_steps=p1_remain,
                    label='Phase 1 MNIST',
                    step_offset=p1_anchor + p1_done)
    else:
        print(f"[Phase 1] training already complete ({p1_done}/{p1_budget}), skipping to eval")

    acc_mnist_initial = eval_phase(field, encoder, te,
                                    class_offset=MNIST_RANGE[0],
                                    n_classes_in_phase=10,
                                    label='MNIST after Phase 1')

    if not _phase_already_complete(1):
        consolidate(field, MNIST_RANGE[0], 10, 'Phase 1 MNIST')
        log_event('phase_1_complete', acc_mnist=acc_mnist_initial,
                  experience_count=field.experience_count)
        sentry_event('phase_1_complete', acc_mnist=acc_mnist_initial)

    field.save_checkpoint(CKPT_LIFETIME)
    return acc_mnist_initial

def run_phase_2():
    """CIFAR-100 continual learning. Train 30K on CIFAR, eval CIFAR, re-eval MNIST."""
    field = load_or_build_brain()
    encoder = UniversalEncoder()
    cf_tr, cf_te = load_cifar100()
    mn_tr, mn_te = load_mnist()

    # Bring CIFAR's teach slots online (MNIST 0-9 stay alive/locked from Phase 1).
    # TinyImageNet's 200 slots remain dormant until Phase 3.
    set_active_teach_slots(field, CIFAR_RANGE[0] + 100)

    # Get pre-CIFAR MNIST baseline
    acc_mnist_before_cifar = eval_phase(field, encoder, mn_te,
                                         class_offset=MNIST_RANGE[0],
                                         n_classes_in_phase=10,
                                         label='MNIST before CIFAR')

    p2_anchor = phase_anchor(2)
    p2_budget = CONFIG['phase2_cifar_steps']
    p2_done   = max(0, field.experience_count - p2_anchor)
    p2_remain = max(0, p2_budget - p2_done)
    if p2_remain > 0:
        print(f"[Phase 2] resuming from step {p2_done}/{p2_budget} — {p2_remain} steps remaining")
        train_phase(field, encoder, cf_tr,
                    class_offset=CIFAR_RANGE[0],
                    n_classes_in_phase=100,
                    n_steps=p2_remain,
                    label='Phase 2 CIFAR-100',
                    step_offset=p2_anchor + p2_done)
    else:
        print(f"[Phase 2] training already complete ({p2_done}/{p2_budget}), skipping to eval")

    acc_cifar = eval_phase(field, encoder, cf_te,
                           class_offset=CIFAR_RANGE[0],
                           n_classes_in_phase=100,
                           label='CIFAR after Phase 2')

    acc_mnist_after_cifar = eval_phase(field, encoder, mn_te,
                                        class_offset=MNIST_RANGE[0],
                                        n_classes_in_phase=10,
                                        label='MNIST RETENTION after CIFAR')

    retention = (acc_mnist_after_cifar / max(acc_mnist_before_cifar, 1)) * 100
    if not _phase_already_complete(2):
        consolidate(field, CIFAR_RANGE[0], 100, 'Phase 2 CIFAR-100')
        log_event('phase_2_complete',
                  acc_mnist_before_cifar=acc_mnist_before_cifar,
                  acc_cifar=acc_cifar,
                  acc_mnist_after_cifar=acc_mnist_after_cifar,
                  mnist_retention_pct=retention,
                  experience_count=field.experience_count)
        sentry_event('phase_2_complete',
                     acc_mnist_before_cifar=acc_mnist_before_cifar,
                     acc_cifar=acc_cifar,
                     acc_mnist_after_cifar=acc_mnist_after_cifar,
                     mnist_retention_pct=retention)

    field.save_checkpoint(CKPT_LIFETIME)
    return acc_cifar, acc_mnist_after_cifar

def run_phase_3():
    """Tiny ImageNet continual learning. Train, eval TI + CIFAR + MNIST retention."""
    field = load_or_build_brain()
    encoder = UniversalEncoder()
    ti_tr, ti_te = load_tiny_imagenet()
    cf_tr, cf_te = load_cifar100()
    mn_tr, mn_te = load_mnist()

    # Final phase brings TinyImageNet's 200 slots online → all 310 now active.
    set_active_teach_slots(field, TI_RANGE[0] + 200)

    p3_anchor = phase_anchor(3)
    p3_budget = CONFIG['phase3_ti_steps']
    p3_done   = max(0, field.experience_count - p3_anchor)
    p3_remain = max(0, p3_budget - p3_done)
    if p3_remain > 0:
        print(f"[Phase 3] resuming from step {p3_done}/{p3_budget} — {p3_remain} steps remaining")
        train_phase(field, encoder, ti_tr,
                    class_offset=TI_RANGE[0],
                    n_classes_in_phase=200,
                    n_steps=p3_remain,
                    label='Phase 3 TinyImageNet',
                    step_offset=p3_anchor + p3_done)
    else:
        print(f"[Phase 3] training already complete ({p3_done}/{p3_budget}), skipping to eval")

    acc_ti = eval_phase(field, encoder, ti_te,
                        class_offset=TI_RANGE[0],
                        n_classes_in_phase=200,
                        label='TI after Phase 3')

    acc_cifar = eval_phase(field, encoder, cf_te,
                           class_offset=CIFAR_RANGE[0],
                           n_classes_in_phase=100,
                           label='CIFAR RETENTION after TI')

    acc_mnist = eval_phase(field, encoder, mn_te,
                           class_offset=MNIST_RANGE[0],
                           n_classes_in_phase=10,
                           label='MNIST RETENTION after TI')

    if not _phase_already_complete(3):
        consolidate(field, TI_RANGE[0], 200, 'Phase 3 TinyImageNet')
        log_event('phase_3_complete',
                  acc_ti=acc_ti, acc_cifar_retained=acc_cifar, acc_mnist_retained=acc_mnist,
                  experience_count=field.experience_count)
        sentry_event('phase_3_complete',
                     acc_ti=acc_ti, acc_cifar_retained=acc_cifar, acc_mnist_retained=acc_mnist)

    field.save_checkpoint(CKPT_LIFETIME)
    return acc_ti, acc_cifar, acc_mnist

def run_phase_4():
    """Final MNIST re-evaluation. NO training. Pure memory test."""
    field = load_or_build_brain()
    encoder = UniversalEncoder()
    _, mn_te = load_mnist()
    _, cf_te = load_cifar100()
    _, ti_te = load_tiny_imagenet()

    acc_mnist = eval_phase(field, encoder, mn_te,
                           class_offset=MNIST_RANGE[0],
                           n_classes_in_phase=10,
                           label='MNIST FINAL re-eval')
    acc_cifar = eval_phase(field, encoder, cf_te,
                           class_offset=CIFAR_RANGE[0],
                           n_classes_in_phase=100,
                           label='CIFAR FINAL re-eval')
    acc_ti = eval_phase(field, encoder, ti_te,
                        class_offset=TI_RANGE[0],
                        n_classes_in_phase=200,
                        label='TI FINAL re-eval')

    print(f"\n{'='*60}")
    print(f" LIFETIME CONTINUAL LEARNING — FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  MNIST  (10 cls,  trained Phase 1): {acc_mnist:.2f}%")
    print(f"  CIFAR  (100 cls, trained Phase 2): {acc_cifar:.2f}%")
    print(f"  TI     (200 cls, trained Phase 3): {acc_ti:.2f}%")
    print(f"  Chance: MNIST=10%, CIFAR=1%, TI=0.5%")

    log_event('phase_4_final',
              acc_mnist_final=acc_mnist, acc_cifar_final=acc_cifar, acc_ti_final=acc_ti)
    sentry_event('lifetime_final',
                 acc_mnist=acc_mnist, acc_cifar=acc_cifar, acc_ti=acc_ti)

# ── Log management ────────────────────────────────────────────────────────────

def log_event(event, **data):
    import datetime
    entry = {'event': event, 'ts': datetime.datetime.utcnow().isoformat(), **data}
    log = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            pass
    log.append(entry)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"[log] appended {event}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], required=True,
                        help="Which phase to run: 1=MNIST, 2=CIFAR, 3=TI, 4=Final eval")
    args = parser.parse_args()

    t0 = time.time()
    if args.phase == 1:
        run_phase_1()
    elif args.phase == 2:
        run_phase_2()
    elif args.phase == 3:
        run_phase_3()
    elif args.phase == 4:
        run_phase_4()
    dt = time.time() - t0
    print(f"\n[done] phase {args.phase} took {dt:.0f}s ({dt/60:.1f} min)")

    if SENTRY_OK:
        sentry_sdk.flush(timeout=10)
