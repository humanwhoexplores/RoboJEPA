"""Figures and tables for the report from data/, runs/ and results/.

    python -m robojepa.figures
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
# Fixed method -> color mapping (categorical slots 1-3; random is a neutral reference).
COLORS = {"jepa": "#2a78d6", "direct": "#eb6834", "oracle": "#1baf7a", "random": "#8f8e87"}
LABELS = {"jepa": "JEPA (predicted)", "direct": "Direct classifier", "oracle": "Oracle (true features)", "random": "Random"}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})


def scene_examples():
    f = sorted(glob.glob("data/test_id/scene_*.npz"))[:3]
    if not f:
        return
    names = {-1: "no region", 0: "left", 1: "right"}
    fig, axes = plt.subplots(len(f), 5, figsize=(11, 2.3 * len(f)))
    for r, path in enumerate(f):
        d = np.load(path)
        picks = [np.where(d["region"] == k)[0] for k in (0, 1, -1)]
        picks = [p[0] for p in picks if len(p)][:4]
        ims = [(d["start_img"], f"start  (left={d['colors'][0]})")]
        ims += [(d["imgs"][i, d["n_push"][i] - 1], f"{d['n_push'][i]}-push -> {names[int(d['region'][i])]}") for i in picks]
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c < len(ims):
                ax.imshow(ims[c][0]), ax.set_title(ims[c][1], fontsize=8, color=INK)
    fig.savefig(f"{OUT}/scenes.png"), plt.close(fig)


def dataset_stats():
    lines = ["| split | scenes | candidates | 1-push | 2-push | none | left | right | scenes w/ success |", "|---" * 9 + "|"]
    for sp in ("train", "val", "test_id", "test_ood"):
        fs = sorted(glob.glob(f"data/{sp}/scene_*.npz"))
        if not fs:
            continue
        ds = [np.load(f) for f in fs]
        r = np.concatenate([d["region"] for d in ds])
        n = np.concatenate([d["n_push"] for d in ds])
        any_succ = np.mean([(d["region"] >= 0).any() for d in ds])
        lines.append(f"| {sp} | {len(fs)} | {len(r)} | {(n == 1).sum()} | {(n == 2).sum()} | "
                     + " | ".join(f"{(r == k).mean():.1%}" for k in (-1, 0, 1)) + f" | {any_succ:.0%} |")
    open(f"{OUT}/dataset_stats.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _results():
    res = {}
    for p in glob.glob("results/n*/results.json"):
        res[int(p.split("/n")[1].split("/")[0])] = json.load(open(p))
    return dict(sorted(res.items()))


def main_bars(res, n):
    if n not in res:
        return
    groups = ["all", "1push_candidates", "2push_candidates", "position_instr", "color_instr", "heldout_paraphrases"]
    gl = ["All", "1-push cands", "2-push cands", "Position instr.", "Color instr.", "Held-out paraphr."]
    splits = [s for s in ("test_id", "test_ood") if s in res[n]]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    methods = ["random", "direct", "jepa", "oracle"]
    w = 0.2
    for ax, sp in zip(axes, splits):
        v = res[n][sp]["selection_success_solvable"]
        x = np.arange(len(groups))
        for j, m in enumerate(methods):
            ax.bar(x + (j - 1.5) * w, [v[g][m] for g in groups], w * 0.9, color=COLORS[m], label=LABELS[m])
        ax.set_xticks(x, gl, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{sp}  (trained on {n} scenes)", color=INK, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Selection success (solvable pairs)")
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.2), fontsize=9)
    fig.savefig(f"{OUT}/main_comparison.png"), plt.close(fig)


def sweep(res):
    if len(res) < 2:
        return
    ns = list(res)
    splits = [s for s in ("test_id", "test_ood") if s in res[ns[0]]]
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, splits):
        for m in ("random", "direct", "jepa", "oracle"):
            y = [res[n][sp]["selection_success_solvable"]["all"][m] for n in ns]
            ax.plot(ns, y, "-o", color=COLORS[m], lw=2, ms=6, mec="#fcfcfb", mew=1.5)
            ax.annotate(LABELS[m], (ns[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK)
        ax.set_xscale("log"), ax.set_xticks(ns, [str(n) for n in ns]), ax.minorticks_off()
        ax.set_xlabel("Training scenes"), ax.set_title(sp, color=INK, fontsize=10), ax.set_ylim(0, 1)
    axes[0].set_ylabel("Selection success (solvable pairs)")
    fig.savefig(f"{OUT}/data_efficiency.png"), plt.close(fig)
    lines = ["| scenes | split | random | direct | jepa | oracle |", "|---|---|---|---|---|---|"]
    for n in ns:
        for sp in splits:
            v = res[n][sp]["selection_success_solvable"]["all"]
            lines.append(f"| {n} | {sp} | " + " | ".join(f"{v[m]:.3f}" for m in ("random", "direct", "jepa", "oracle")) + " |")
    open(f"{OUT}/data_efficiency.md", "w").write("\n".join(lines) + "\n")


def wm_error(res, n):
    if n not in res:
        return
    splits = [s for s in ("test_id", "test_ood") if s in res[n]]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels, pred, copy = [], [], []
    for sp in splits:
        for k, v in res[n][sp]["world_model_mse"].items():
            labels.append(f"{sp}\n{k}"), pred.append(v["pred_mse"]), copy.append(v["copy_mse"])
    x = np.arange(len(labels))
    ax.bar(x - 0.2, copy, 0.36, color=COLORS["random"], label="No-change baseline (start features)")
    ax.bar(x + 0.2, pred, 0.36, color=COLORS["jepa"], label="Predictor rollout")
    ax.set_xticks(x, labels), ax.set_ylabel("Feature MSE vs. true terminal"), ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(f"{OUT}/world_model_error.png"), plt.close(fig)


def curves(n):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, m in zip(axes, ("predictor", "scorer", "direct")):
        p = f"runs/n{n}/{m}/log.json"
        if not os.path.exists(p):
            continue
        log = json.load(open(p))["log"]
        ep = [r["step"] for r in log]
        ax.plot(ep, [r["train_loss"] for r in log], color=COLORS["random"], lw=2, label="train")
        ax.plot(ep, [r["val_loss"] for r in log], color=COLORS["jepa"], lw=2, label="val")
        ax.set_title(m, color=INK, fontsize=10), ax.set_xlabel("step"), ax.legend(fontsize=8)
    axes[0].set_ylabel("loss (MSE / weighted BCE)")
    fig.savefig(f"{OUT}/learning_curves.png"), plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    scene_examples()
    dataset_stats()
    res = _results()
    n = max(res) if res else 400
    main_bars(res, n), sweep(res), wm_error(res, n), curves(n)
    print(f"wrote {sorted(os.listdir(OUT))}")


if __name__ == "__main__":
    main()
