"""
UERF CHAT — a character-level LANGUAGE MODEL on the coupled-oscillator field.
No neural network, no backprop. A chatbot is next-token prediction; here each
"class" is a character, so the associative UERF field IS the language model:
  context (last K chars) --encode--> field input --relax--> teach-slot per char
  --argmax/sample--> next char --append--> repeat  => generated text.

Trains online over a real text corpus (physics, not gradient descent), reports
next-char accuracy + generated samples periodically, checkpoints the brain.
"""
import os, sys, math, time, glob, urllib.request, numpy as np, torch

def load_brain():
    p = sorted(glob.glob("/kaggle/input/**/uerf_brain.py", recursive=True), key=len)
    if p:
        sys.path.insert(0, os.path.dirname(p[0]))
    import uerf_brain as U
    return U

def get_corpus():
    urls = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    ]
    for u in urls:
        try:
            txt = urllib.request.urlopen(u, timeout=20).read().decode("utf-8", "ignore")
            if len(txt) > 5000:
                print(f"corpus: fetched {len(txt)} chars from {u}", flush=True)
                return txt
        except Exception as e:
            print("corpus fetch failed:", e, flush=True)
    # fallback: embedded sample (small, still exercises the pipeline)
    return ("To be, or not to be, that is the question: whether tis nobler in the "
            "mind to suffer the slings and arrows of outrageous fortune, or to take "
            "arms against a sea of troubles and by opposing end them. ") * 40


class CharCtx:
    """Encode the last K chars into an n_sensory field input. Fixed random
    projection of position-weighted char one-hots — recent chars weighted more.
    No learned params (a fixed sensory transduction, like the MNIST projection)."""
    def __init__(self, vocab, K=24, n_sensory=96, seed=42, dev="cuda"):
        self.V, self.K, self.n, self.dev = vocab, K, n_sensory, dev
        g = torch.Generator().manual_seed(seed)
        self.W = (torch.randn(K * vocab, n_sensory, generator=g) / math.sqrt(K * vocab)).to(dev)
        self.decay = torch.tensor([0.9 ** (K - 1 - j) for j in range(K)], device=dev)

    def __call__(self, ctx_idxs):
        v = torch.zeros(self.K, self.V, device=self.dev)
        for j, ci in enumerate(ctx_idxs):
            if ci >= 0:
                v[j, ci] = self.decay[j]
        x = v.reshape(-1) @ self.W
        return x / x.norm().clamp(min=1e-6)


def main():
    U = load_brain()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(369)

    text = get_corpus()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_tr = int(0.9 * len(data)); train, val = data[:n_tr], data[n_tr:]
    print(f"vocab={V} chars, corpus={len(data)} ({len(train)} train / {len(val)} val)", flush=True)

    K = 24
    enc = CharCtx(V, K=K, n_sensory=96, seed=42, dev=dev)
    f = U.UERFField(n_max=600, n_initial=350, d=32, input_dim=96, n_classes=V + 2)
    f.start_emergent()
    slot = {}                      # char idx -> teach slot

    def ctx_at(seq, i):
        lo = max(0, i - K)
        c = seq[lo:i].tolist()
        return [-1] * (K - len(c)) + c

    def teach_step(seq, i, n_relax=4):
        target = int(seq[i]); x = enc(ctx_at(seq, i))
        if target not in slot:
            sig = (x.unsqueeze(-1) * f.c[:f.input_dim]).sum(0)
            slot[target] = f.birth_category(sig)
        tv = torch.zeros(f.n_classes, device=dev); tv[slot[target]] = 1.0
        f.experience(x, teaching_vector=tv, n_relax=n_relax, domain_lo=0, domain_hi=f.n_classes)

    def acc(n=400):
        if not slot: return 0.0
        inv = {v: k for k, v in slot.items()}
        lo, hi = min(slot.values()), max(slot.values()) + 1
        rng = np.random.RandomState(1); c = t = 0
        for _ in range(n):
            i = rng.randint(K, len(val))
            p = f.eval_predict(enc(ctx_at(val, i)), n_relax=6, domain_lo=lo, domain_hi=hi)
            c += int(inv.get(p) == int(val[i])); t += 1
        return 100.0 * c / max(t, 1)

    def generate(seed, n=220, temp=0.75):
        inv = {v: k for k, v in slot.items()}
        lo, hi = min(slot.values()), max(slot.values()) + 1
        ctx = [stoi.get(ch, -1) for ch in seed][-K:]
        ctx = [-1] * (K - len(ctx)) + ctx
        out = seed
        for _ in range(n):
            _, sc = f.eval_predict(enc(ctx), n_relax=6, domain_lo=lo, domain_hi=hi, return_scores=True)
            if sc is None: break
            logits = torch.full((V,), -1e9, device=dev)
            for s_idx in range(lo, hi):
                if inv.get(s_idx) is not None:
                    logits[inv[s_idx]] = sc[s_idx]
            p = torch.softmax(logits / temp, 0)
            nxt = int(torch.multinomial(p, 1))
            out += itos[nxt]; ctx = ctx[1:] + [nxt]
        return out

    # ── EFFICIENT online training: ONE (or few) passes over a MODEST corpus ──
    # The brain's claim is data efficiency — it should pick up language from
    # limited exposure, not brute-force. So we stream a bounded slice of the
    # corpus a few times and watch structure emerge, exactly like the few-shot
    # digit runs. If it needed overnight brute training, that would DISPROVE the
    # efficiency claim.
    N_TRAIN  = int(os.environ.get("UERF_N_TRAIN",  20000))   # positions seen total
    REPORT_EVERY = int(os.environ.get("UERF_REPORT", 2500))
    # sample positions from the first slice of the corpus (limited exposure)
    slice_end = min(len(train), int(os.environ.get("UERF_SLICE", 40000)))
    rng = np.random.RandomState(7)
    t0 = time.time()
    print("=" * 64); print(" UERF CHAT — char LM on the oscillator field (efficient, no brute force)")
    print(f" seeing ~{N_TRAIN} positions from a {slice_end}-char slice"); print("=" * 64, flush=True)
    for step in range(1, N_TRAIN + 1):
        i = rng.randint(K, slice_end)
        teach_step(train, i, n_relax=4)
        if step % REPORT_EVERY == 0:
            a = acc(); rate = step / (time.time() - t0)
            print(f"\n[seen {step} | {rate:.1f}/s | {len(slot)}/{V} chars | val next-char {a:.1f}% (chance {100/V:.1f})]", flush=True)
            print("  SAMPLE:", repr(generate("The ", n=160)), flush=True)
            try:
                f.save_checkpoint("/kaggle/working/uerf_chat.pt")
            except Exception as e:
                print("  ckpt err:", e, flush=True)

    print("\n" + "=" * 64); print(" FINAL"); print("=" * 64, flush=True)
    print(f"trained {step} steps, {len(slot)}/{V} chars, val next-char acc {acc(800):.1f}% (chance {100/V:.2f})", flush=True)
    for seed in ["The ", "My lord", "What "]:
        print(f"\n  seed {seed!r}:\n  {generate(seed, n=300)!r}", flush=True)
    try:
        f.save_checkpoint("/kaggle/working/uerf_chat.pt"); print("\nsaved /kaggle/working/uerf_chat.pt", flush=True)
    except Exception as e:
        print("save err:", e, flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
