"""
plot_hp_features.py
-------------------
Loads the trained SAE checkpoint and plots the Top-20 Harry Potter-specific
features by specificity ratio (target_freq / neutral_freq).

Requires:
  - checkpoints/sae_layer_12.pt   (from  python src/sae/train.py)
  - src/data/target_corpus.txt    (from  python src/data/preprocess.py)

Usage:
  python plot_hp_features.py
  python plot_hp_features.py --layer 12 --top_n 20 --out hp_features.png
"""

import os
import sys
import argparse

import torch
import einops
import matplotlib
matplotlib.use("Agg")          # Safe for headless; change to "TkAgg" if you want a popup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer
from datasets import load_dataset

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.sae.model import TopKSAE


# ── helpers ──────────────────────────────────────────────────────────────────

def get_feature_stats(tokens, model, sae, layer, device, batch_size=8, ctx_len=128):
    """
    Run tokens through GPT-2 → SAE and return per-feature
    (activation_frequency, mean_activation_when_firing).
    """
    d_sae = sae.d_sae
    fire_count = torch.zeros(d_sae, device=device)
    act_sum    = torch.zeros(d_sae, device=device)
    total      = 0

    chunks = [tokens[i : i + ctx_len] for i in range(0, len(tokens) - ctx_len + 1, ctx_len)]
    batches = []
    for i in range(0, len(chunks), batch_size):
        b = chunks[i : i + batch_size]
        if b:
            t = torch.tensor(b, device=device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.shape[1] == ctx_len:
                batches.append(t)

    with torch.no_grad():
        for batch in tqdm(batches, desc="  batches", leave=False):
            _, cache = model.run_with_cache(batch, stop_at_layer=layer + 1)
            acts = cache[f"blocks.{layer}.hook_resid_post"]
            _, z_sparse = sae(acts)                          # (b, s, d_sae)
            z_flat = einops.rearrange(z_sparse, "b s d -> (b s) d")

            fire_count += (z_flat > 0).float().sum(dim=0)
            act_sum    += z_flat.sum(dim=0)
            total      += z_flat.shape[0]

    freq      = fire_count / total
    mean_act  = act_sum / (fire_count + 1e-8)
    return freq, mean_act


def compute_top_features(args, device):
    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading {args.model_name} …")
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    model.eval()

    # ── Load SAE ────────────────────────────────────────────────────────────
    ckpt = f"checkpoints/sae_layer_{args.layer}.pt"
    if not os.path.exists(ckpt):
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt}\n"
                 f"  Train first:  python src/sae/train.py --layer {args.layer}")

    d_model = model.cfg.d_model
    d_sae   = d_model * args.expansion_factor
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k)
    sae.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    sae.to(device).eval()
    print(f"SAE loaded  (d_model={d_model}, d_sae={d_sae}, k={args.k})")

    # ── Target corpus (Harry Potter) ─────────────────────────────────────────
    if not os.path.exists(args.target_corpus):
        sys.exit(f"[ERROR] Target corpus not found: {args.target_corpus}\n"
                 f"  Preprocess first:  python src/data/preprocess.py")

    print("Tokenising Harry Potter corpus …")
    with open(args.target_corpus, encoding="utf-8") as f:
        hp_text = f.read()
    hp_tokens = model.tokenizer.encode(hp_text)[: args.max_tokens]

    # ── Neutral corpus (WikiText-2 + TinyStories) ────────────────────────────
    print("Loading neutral corpus (WikiText-2) …")
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_texts = [t for t in wiki["text"] if t.strip()]

    fiction_texts: list[str] = []
    try:
        print("Loading TinyStories (2 000 samples) …")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        it = iter(ds)
        for _ in range(2000):
            fiction_texts.append(next(it)["text"])
    except Exception as e:
        print(f"  Warning: TinyStories unavailable ({e}). Using WikiText-2 only.")

    neutral_text  = "\n".join(wiki_texts + fiction_texts)
    neutral_tokens = model.tokenizer.encode(neutral_text)[: args.max_tokens]

    # ── Compute stats ────────────────────────────────────────────────────────
    print("\nComputing HP feature stats …")
    hp_freq,  hp_mean  = get_feature_stats(hp_tokens,      model, sae, args.layer, device)
    print("Computing neutral feature stats …")
    neu_freq, neu_mean = get_feature_stats(neutral_tokens, model, sae, args.layer, device)

    # ── Specificity ratio + filtering ────────────────────────────────────────
    ratio = hp_freq / (neu_freq + 1e-6)

    valid = (
        (hp_freq  >  0.001) &          # must fire on HP text
        (neu_freq <  0.005) &          # must not be generic
        (hp_freq  <  0.2)   &          # not a grammar feature
        (ratio    >= args.min_ratio)   # significantly HP-specific
    )
    metric = ratio.clone()
    metric[~valid] = -float("inf")

    top_vals, top_inds = torch.topk(metric, k=args.top_n)

    return (
        top_inds.cpu(),
        top_vals.cpu(),
        hp_freq[top_inds].cpu(),
        neu_freq[top_inds].cpu(),
        ratio[top_inds].cpu(),
        hp_mean[top_inds].cpu(),
    )


# ── plotting ─────────────────────────────────────────────────────────────────

def plot(top_inds, top_vals, hp_freq, neu_freq, ratio, hp_mean, args):
    n = len(top_inds)
    labels = [f"F{idx.item()}" for idx in top_inds]
    x      = np.arange(n)
    width  = 0.35

    # ── colour gradient by ratio ──────────────────────────────────────────────
    norm   = plt.Normalize(ratio.min().item(), ratio.max().item())
    cmap   = plt.cm.plasma
    colours = [cmap(norm(r.item())) for r in ratio]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 1.4]},
        facecolor="#0d1117",
    )
    fig.suptitle(
        f"Top-{n} HP-Specific SAE Features  ·  Layer {args.layer}  ·  {args.model_name}",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )

    # ── TOP PANEL: grouped bar chart (activation frequencies) ─────────────────
    ax = axes[0]
    ax.set_facecolor("#161b22")

    bars_hp  = ax.bar(x - width / 2, hp_freq.numpy(),  width, color=colours,
                      alpha=0.92, label="HP freq")
    bars_neu = ax.bar(x + width / 2, neu_freq.numpy(), width, color="gray",
                      alpha=0.50, label="Neutral freq")

    # annotate ratio on top of each HP bar
    for i, (bar, r) in enumerate(zip(bars_hp, ratio)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003,
            f"{r.item():.0f}×",
            ha="center", va="bottom", fontsize=7.5, color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, color="white")
    ax.set_ylabel("Activation Frequency", color="white")
    ax.set_ylim(0, max(hp_freq.max().item() * 1.25, 0.02))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(
        handles=[
            mpatches.Patch(color=cmap(0.9), label="HP activation freq"),
            mpatches.Patch(color="gray",    label="Neutral activation freq"),
        ],
        facecolor="#0d1117", labelcolor="white", fontsize=9,
    )
    ax.grid(axis="y", color="#30363d", linewidth=0.6)

    # ── BOTTOM PANEL: mean activation when firing ─────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")

    bar_colours = [cmap(norm(r.item())) for r in ratio]
    ax2.bar(x, hp_mean.numpy(), color=bar_colours, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5, color="white")
    ax2.set_ylabel("Mean Activation\n(when firing)", color="white", fontsize=9)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")
    ax2.grid(axis="y", color="#30363d", linewidth=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("Specificity Ratio (HP / Neutral)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(rect=[0, 0, 0.97, 0.96])
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {args.out}")
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot top HP-specific SAE features")
    parser.add_argument("--layer",            type=int,   default=12)
    parser.add_argument("--model_name",       type=str,   default="gpt2-medium")
    parser.add_argument("--expansion_factor", type=int,   default=16)
    parser.add_argument("--k",                type=int,   default=32)
    parser.add_argument("--top_n",            type=int,   default=20)
    parser.add_argument("--max_tokens",       type=int,   default=200_000)
    parser.add_argument("--min_ratio",        type=float, default=20.0)
    parser.add_argument("--target_corpus",    type=str,   default="src/data/target_corpus.txt")
    parser.add_argument("--out",              type=str,   default="hp_features_top20.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    top_inds, top_vals, hp_freq, neu_freq, ratio, hp_mean = compute_top_features(args, device)

    print(f"\n{'Feature':>10}  {'Ratio':>10}  {'HP Freq':>10}  {'Neu Freq':>10}  {'Mean Act':>10}")
    print("-" * 57)
    for i in range(len(top_inds)):
        print(f"F{top_inds[i].item():>9}  {ratio[i].item():>10.1f}  "
              f"{hp_freq[i].item():>10.4f}  {neu_freq[i].item():>10.4f}  "
              f"{hp_mean[i].item():>10.4f}")

    plot(top_inds, top_vals, hp_freq, neu_freq, ratio, hp_mean, args)


if __name__ == "__main__":
    main()
