import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/claude/cace/outputs"
FIG = f"{OUT}/figures"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                      "font.family": "serif", "axes.grid": True, "grid.alpha": 0.25})
COLORS = {"CACE": "#1a5276", "LinearSSM": "#d68910", "SmallNeuralCDE": "#c0392b", "GRUD": "#27ae60"}
LABELS = {"CACE": "CACE (proposed)", "LinearSSM": "Linear SSM (S4/Mamba-family)",
          "SmallNeuralCDE": "Neural CDE", "GRUD": "GRU-D"}


def fig_main_results():
    with open(f"{OUT}/metrics/main_results.json") as f:
        r = json.load(f)
    names = list(r.keys())
    aurocs = [r[n]["test_auroc"] for n in names]
    auprcs = [r[n]["test_auprc"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = [COLORS[n] for n in names]
    axes[0].bar(names, aurocs, color=colors)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    axes[0].set_ylabel("Test AUROC"); axes[0].set_title("In-hospital mortality prediction")
    axes[0].set_ylim(0.4, 0.9); axes[0].legend(fontsize=8)
    axes[1].bar(names, auprcs, color=colors)
    axes[1].set_ylabel("Test AUPRC"); axes[1].set_title("(construct-validity check, not a\npre-registered hypothesis)")
    for ax in axes:
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_main_task_performance.png")
    plt.close(fig)


def fig_h1():
    with open(f"{OUT}/metrics/h1_order_invariance.json") as f:
        r = json.load(f)
    names = [n for n in r if "diagnosis" in r[n]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(names))
    within_cap = [max(r[n]["diagnosis"]["max_delta_within_cap_only"], 1e-7) for n in names]
    exceeding_cap = [r[n]["diagnosis"]["max_delta_exceeding_cap_only"] for n in names]
    ax.bar(x - 0.18, within_cap, width=0.35, color="#1a5276",
           label="Max |output change| -- patients within the 600-event cap\n(isolates the actual cross-channel-order claim)")
    ax.bar(x + 0.18, exceeding_cap, width=0.35, color="#aeb6bf",
           label="Max |output change| -- patients EXCEEDING the cap\n(truncation-boundary artifact, not an order-invariance failure)")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in names], fontsize=9)
    ax.axhline(1e-5, color="red", linestyle="--", linewidth=1, label="Falsification threshold (1e-5)")
    ax.set_ylabel("Max |change in model output| under\ncross-channel same-timestamp reordering (symlog scale)")
    ax.set_title("H1: Cross-channel order invariance\n(CACE: exactly 0.0 within the event cap -- bit-for-bit invariant)")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_h1_order_invariance.png")
    plt.close(fig)


def fig_h2():
    with open(f"{OUT}/metrics/h2_robustness.json") as f:
        r = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name in r["dropout"]:
        axes[0].plot(r["drop_fracs"], r["dropout"][name], marker="o", color=COLORS[name], label=LABELS[name])
        axes[1].plot(r["jitter_stds"], r["jitter"][name], marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].set_xlabel("Fraction of events randomly dropped")
    axes[0].set_ylabel("Test AUROC")
    axes[0].set_title("H2a: Robustness to observation sparsification")
    axes[1].set_xlabel("Timestamp jitter, Gaussian std (minutes)")
    axes[1].set_ylabel("Test AUROC")
    axes[1].set_title("H2b: Robustness to timing jitter")
    for ax in axes:
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_h2_robustness.png")
    plt.close(fig)


def fig_h3():
    with open(f"{OUT}/metrics/h3_scaling.json") as f:
        r = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for name, d in r.items():
        if name == "diagnosis":
            continue
        axes[0].plot(d["lengths"], d["times_sec"], marker="o", color=COLORS[name], label=LABELS[name])
        axes[1].plot(d["lengths"], d["per_step_ms"], marker="o", color=COLORS[name], label=LABELS[name])
        axes[2].plot(d["lengths"], d["peak_mem_mb"], marker="o", color=COLORS[name], label=LABELS[name])
    for ax, title, ylabel in zip(
        axes,
        ["Total forward-pass time", "Per-event time (time / L)", "Peak memory"],
        ["seconds", "milliseconds / event", "MB"],
    ):
        ax.set_xlabel("Sequence length L (events)")
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xscale("log"); ax.legend(fontsize=7.5)
    fig.suptitle("H3: Computational scaling with sequence length")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_h3_scaling.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_main_results()
    fig_h1()
    fig_h2()
    fig_h3()
    print("All figures written to", FIG)
