"""
EN3150 Assignment 03 - Q3: Optimizer selection and tuning.

Trains Model B (depthwise separable CNN) with different optimizers and keeps
everything else identical, so the only thing that changes between runs is the
optimizer:

    same data split     (prepare_data.load_datasets, seed 42)
    same initial weights (keras.utils.set_random_seed(SEED) before each build)
    same batch order    (datasets reloaded with the same shuffle seed per run)
    same epochs, batch size, loss and model

Runs:
    adam          Adam,            lr = 1e-3   (our chosen optimizer)
    sgd           plain SGD,       lr = 1e-2
    sgd_m0.9      SGD + momentum,  lr = 1e-2, momentum = 0.9
  with --sweep, also:
    sgd_m0.5      SGD + momentum,  lr = 1e-2, momentum = 0.5
    sgd_m0.99     SGD + momentum,  lr = 1e-2, momentum = 0.99

Only the validation set is used here. The test set stays untouched until Q4.

Usage:
    python train_optimizers.py                # 3 main runs, 20 epochs
    python train_optimizers.py --sweep        # + momentum 0.5 and 0.99
    python train_optimizers.py --epochs 30
    python train_optimizers.py --plot-only    # redraw plots from saved results

Outputs (in results/q3/):
    history_<run>.json      per-epoch loss/accuracy + time per epoch
    q3_curves.png           loss and accuracy curves, all runs on shared axes
    q3_momentum.png         momentum sweep only (if --sweep was run)
    q3_summary.csv / .md    table for the report
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save figures without opening a window
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
from tensorflow import keras

from models import build_model_b
from prepare_data import load_datasets

SEED = 42
BATCH_SIZE = 32
OUT_DIR = Path("results/q3")

# name -> (label for plots, optimizer factory)
# Factories (lambdas) so each run gets a fresh optimizer with fresh state.
RUNS = {
    "adam":      ("Adam (lr=1e-3)",
                  lambda: keras.optimizers.Adam(learning_rate=1e-3)),
    "sgd":       ("SGD (lr=1e-2)",
                  lambda: keras.optimizers.SGD(learning_rate=1e-2)),
    "sgd_m0.9":  ("SGD + momentum 0.9 (lr=1e-2)",
                  lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)),
}
SWEEP_RUNS = {
    "sgd_m0.5":  ("SGD + momentum 0.5 (lr=1e-2)",
                  lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.5)),
    "sgd_m0.99": ("SGD + momentum 0.99 (lr=1e-2)",
                  lambda: keras.optimizers.SGD(learning_rate=1e-2, momentum=0.99)),
}
ALL_RUNS = {**RUNS, **SWEEP_RUNS}

# Fixed colour per run (colour follows the run, not its position in a plot).
COLORS = {
    "adam": "#2a78d6",       # blue
    "sgd": "#eb6834",        # orange
    "sgd_m0.9": "#1baf7a",   # aqua
    "sgd_m0.5": "#eda100",   # yellow
    "sgd_m0.99": "#e87ba4",  # magenta
}


class EpochTimer(keras.callbacks.Callback):
    """Records wall-clock time of every epoch (needed again in Q4)."""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t0)


def train_one(name, epochs):
    label, make_optimizer = ALL_RUNS[name]
    print(f"\n=== {name}: {label} ===")

    # Same seed before building -> identical initial weights for every run.
    keras.utils.set_random_seed(SEED)
    model = build_model_b()
    model.compile(
        optimizer=make_optimizer(),
        loss="sparse_categorical_crossentropy",   # labels are integer class IDs
        metrics=["accuracy"],
    )

    # Reload so each run sees the same shuffled batch order.
    train_ds, val_ds, _ = load_datasets(batch_size=BATCH_SIZE, seed=SEED)

    timer = EpochTimer()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=[timer], verbose=2)

    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h["epoch_time_s"] = timer.times
    h["label"] = label
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"history_{name}.json").write_text(json.dumps(h, indent=2))
    return h


def load_histories(names):
    out = {}
    for n in names:
        p = OUT_DIR / f"history_{n}.json"
        if p.exists():
            out[n] = json.loads(p.read_text())
    return out


def plot_curves(histories, filename, title):
    """Left: loss, right: accuracy. Solid = training, dashed = validation."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        ep = np.arange(1, len(h["loss"]) + 1)
        c = COLORS[name]
        ax_loss.plot(ep, h["loss"], color=c, lw=2, label=f"{h['label']} - train")
        ax_loss.plot(ep, h["val_loss"], color=c, lw=2, ls="--", label=f"{h['label']} - val")
        ax_acc.plot(ep, h["accuracy"], color=c, lw=2)
        ax_acc.plot(ep, h["val_accuracy"], color=c, lw=2, ls="--")

    for ax, ylabel in ((ax_loss, "Cross-entropy loss"), (ax_acc, "Accuracy")):
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#dddddd", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    ax_loss.set_title("Loss")
    ax_acc.set_title("Accuracy")
    fig.suptitle(title)
    fig.legend(*ax_loss.get_legend_handles_labels(), loc="lower center",
               ncol=len(histories), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"saved {OUT_DIR / filename}")


def write_summary(histories):
    header = ["run", "optimizer", "best val acc", "best epoch",
              "final train loss", "final val loss", "final val acc", "mean s/epoch"]
    rows = []
    for name, h in histories.items():
        best = int(np.argmax(h["val_accuracy"]))
        rows.append([
            name, h["label"],
            f"{h['val_accuracy'][best]:.4f}", str(best + 1),
            f"{h['loss'][-1]:.4f}", f"{h['val_loss'][-1]:.4f}",
            f"{h['val_accuracy'][-1]:.4f}",
            # skip epoch 1 (includes graph tracing overhead)
            f"{np.mean(h['epoch_time_s'][1:] or h['epoch_time_s']):.1f}",
        ])
    (OUT_DIR / "q3_summary.csv").write_text(
        "\n".join(",".join(r) for r in [header] + rows) + "\n")
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (OUT_DIR / "q3_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--sweep", action="store_true", help="also run momentum 0.5 and 0.99")
    ap.add_argument("--only", nargs="+", choices=list(ALL_RUNS), help="run just these")
    ap.add_argument("--plot-only", action="store_true", help="skip training, redraw from saved JSON")
    args = ap.parse_args()

    names = args.only or (list(ALL_RUNS) if args.sweep else list(RUNS))
    if not args.plot_only:
        for n in names:
            train_one(n, args.epochs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_h = load_histories(RUNS)
    if main_h:
        plot_curves(main_h, "q3_curves.png",
                    "Q3: Model B - Adam vs SGD vs SGD + momentum")
    sweep_h = load_histories(["sgd", "sgd_m0.5", "sgd_m0.9", "sgd_m0.99"])
    if len(sweep_h) > 2:
        plot_curves(sweep_h, "q3_momentum.png",
                    "Q3: effect of momentum (SGD, lr=1e-2)")
    write_summary(load_histories(ALL_RUNS))


if __name__ == "__main__":
    main()
