"""
EN3150 Assignment 03 - Q4: Custom model training and evaluation.

Trains Model A (standard CNN) and Model B (depthwise separable CNN) with the
optimizer chosen in Q3 (Adam, lr = 1e-3), then evaluates both on the TEST set.

Training setup (identical for A and B):
    - same data split (prepare_data.load_datasets, seed 42)
    - same seed before building each model
    - Adam lr = 1e-3, batch 32, sparse categorical cross-entropy
    - 30 epochs (assignment minimum is 20)
    - data augmentation on the TRAINING set only: random horizontal flip,
      rotation (+/-10%) and zoom (+/-10%). Q3 showed overfitting starting
      around epoch 13; augmentation reduces it. Augmentation is applied in
      the input pipeline, not inside the model, so the saved model and its
      parameter count are exactly the Q2 architecture.
    - learning-rate schedule: ReduceLROnPlateau halves the learning rate when
      validation loss has not improved for 3 epochs (floor 1e-5), starting
      from epoch 9 (earlier epochs are skipped because BatchNorm's lagging
      running statistics make validation loss rise at the start). The first
      Q4 run at a constant lr = 1e-3 gave a very jumpy validation curve for
      Model A; smaller steps late in training settle it. Disable with
      --no-lr-schedule to reproduce the constant-lr run.
    - the weights from the epoch with the best VALIDATION accuracy are kept
      (ModelCheckpoint). The test set plays no part in choosing them.

Evaluation (test set, first and only use):
    accuracy, confusion matrix, per-class precision and recall, macro average.

Comparison table (Q4):
    trainable params, MACs, size on disk (KB), median time per epoch, test accuracy.

Usage:
    python train_eval.py                  # train + evaluate A and B
    python train_eval.py --models B       # just one model
    python train_eval.py --epochs 40
    python train_eval.py --eval-only      # re-evaluate saved models, no training
    python train_eval.py --no-lr-schedule --out-dir results/q4_constant_lr

Outputs (results/q4/):
    model_A.keras, model_B.keras          best-val weights, inference-only (no optimizer state)
    history_A.json, history_B.json        curves + epoch times
    curves_A.png, curves_B.png            train/val loss and accuracy
    confusion_A.png, confusion_B.png      test confusion matrices
    metrics_A.json, metrics_B.json        all test metrics
    q4_comparison.md / .csv               A vs B table for the report
    q4_per_class.md                       per-class precision / recall table
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models import build_model_a, build_model_b, hand_count
from prepare_data import load_datasets

SEED = 42
BATCH_SIZE = 32
LR = 1e-3
OUT_DIR = Path("results/q4")
BUILDERS = {"A": build_model_a, "B": build_model_b}
COLORS = {"train": "#2a78d6", "val": "#eb6834"}


def class_names(data_dir="data"):
    try:
        return json.loads((Path(data_dir) / "summary.json").read_text())["class_names"]
    except (FileNotFoundError, KeyError):
        return [f"class {i}" for i in range(5)]


# ---------------------------------------------------------------- augmentation
def make_augmenter():
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.1, seed=SEED),     # +/- 0.1 * 360 deg = +/- 36 deg max
        layers.RandomZoom(0.1, seed=SEED),
    ], name="augment")


def get_data(augment=True):
    train_ds, val_ds, test_ds = load_datasets(batch_size=BATCH_SIZE, seed=SEED)
    if augment:
        aug = make_augmenter()
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds, test_ds


# ---------------------------------------------------------------- training
class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t0)


class DelayedReduceLR(keras.callbacks.ReduceLROnPlateau):
    """ReduceLROnPlateau that ignores the first `start_epoch` epochs.

    In the first few epochs validation loss RISES even though training is
    going well, because BatchNorm's running averages lag behind the weights.
    A plain ReduceLROnPlateau would read that as a plateau and cut the
    learning rate far too early.
    """

    def __init__(self, start_epoch=8, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:              # epoch is 0-indexed
            if logs is not None:
                logs["learning_rate"] = float(
                    keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
            return
        super().on_epoch_end(epoch, logs)


def train(name, epochs, augment, lr_schedule=True):
    print(f"\n=== Training Model {name} ===")
    keras.utils.set_random_seed(SEED)
    model = BUILDERS[name]()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    train_ds, val_ds, _ = get_data(augment)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"_best_{name}.weights.h5"
    timer = EpochTimer()
    ckpt = keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True)

    callbacks = [timer, ckpt]
    if lr_schedule:
        # From epoch 9 on: halve the lr when val loss stalls for 3 epochs;
        # never below 1e-5.
        callbacks.append(DelayedReduceLR(
            start_epoch=8, monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-5, verbose=1))

    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=callbacks, verbose=2)

    # Restore best-validation weights, then save an inference-only copy:
    # a freshly built (uncompiled) model has no optimizer state, so the file
    # size reflects what would actually be deployed.
    model.load_weights(ckpt_path)
    deploy = BUILDERS[name]()
    deploy.set_weights(model.get_weights())
    deploy.save(OUT_DIR / f"model_{name}.keras")
    ckpt_path.unlink(missing_ok=True)

    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h["epoch_time_s"] = timer.times
    h["augment"] = augment
    h["lr_schedule"] = lr_schedule
    (OUT_DIR / f"history_{name}.json").write_text(json.dumps(h, indent=2))
    plot_curves(name, h)
    return h


def plot_curves(name, h):
    ep = np.arange(1, len(h["loss"]) + 1)
    best = int(np.argmax(h["val_accuracy"])) + 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, h["loss"], color=COLORS["train"], lw=2, label="training")
    a1.plot(ep, h["val_loss"], color=COLORS["val"], lw=2, label="validation")
    a2.plot(ep, h["accuracy"], color=COLORS["train"], lw=2, label="training")
    a2.plot(ep, h["val_accuracy"], color=COLORS["val"], lw=2, label="validation")
    # epochs at which the learning rate was reduced (if a schedule was used)
    lrs = h.get("learning_rate", [])
    drops = [i + 1 for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1] * 0.999]
    for ax, t in ((a1, "Loss"), (a2, "Accuracy")):
        ax.axvline(best, color="#888888", lw=1, ls=":")
        for d in drops:
            ax.axvline(d, color="#bbbbbb", lw=0.8, ls="--")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="#dddddd", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    a1.set_ylabel("Cross-entropy loss")
    a2.set_ylabel("Accuracy")
    note = "; dashed = lr halved" if drops else ""
    fig.suptitle(f"Model {name}: training curves (dotted = best val epoch {best}, weights kept{note})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"curves_{name}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- evaluation
def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm        # rows = true class, columns = predicted class


def evaluate(name, names):
    model = keras.models.load_model(OUT_DIR / f"model_{name}.keras")
    _, _, test_ds = get_data(augment=False)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    k = len(names)
    cm = confusion(y_true, y_pred, k)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(k), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(k), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(k), where=(precision + recall) > 0)

    h = json.loads((OUT_DIR / f"history_{name}.json").read_text())
    times = h["epoch_time_s"]
    rows = hand_count(name)
    metrics = {
        "model": name,
        "test_accuracy": float(tp.sum() / cm.sum()),
        "per_class": {n: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(cm[i].sum())}
                      for i, n in enumerate(names)},
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": cm.tolist(),
        "trainable_params": sum(r[2] for r in rows),
        "non_trainable_params": sum(r[3] for r in rows),
        "macs": sum(r[4] for r in rows),
        "size_on_disk_kb": os.path.getsize(OUT_DIR / f"model_{name}.keras") / 1024,
        "float32_weights_kb": sum(r[2] + r[3] for r in rows) * 4 / 1024,
        # median, skipping epoch 1 (graph tracing); robust to one slow epoch
        "median_s_per_epoch": float(np.median(times[1:] or times)),
        "epochs_trained": len(times),
        "best_val_epoch": int(np.argmax(h["val_accuracy"])) + 1,
        "best_val_accuracy": float(max(h["val_accuracy"])),
    }
    (OUT_DIR / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(name, cm, names)

    print(f"\nModel {name} - test accuracy {metrics['test_accuracy']:.4f}")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'support':>9}")
    for n, m in metrics["per_class"].items():
        print(f"{n:<12}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['support']:>9}")
    print(f"{'macro avg':<12}{metrics['macro_precision']:>10.3f}{metrics['macro_recall']:>10.3f}")
    return metrics


def plot_confusion(name, cm, names):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    thresh = cm.max() / 2
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#222222", fontsize=10)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"Model {name}: test confusion matrix (acc {acc:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def write_tables(all_metrics, names):
    if not all_metrics:
        return
    header = ["Metric"] + [f"Model {m['model']}" for m in all_metrics]
    lines = [
        ("Trainable parameters", lambda m: f"{m['trainable_params']:,}"),
        ("MACs per image", lambda m: f"{m['macs'] / 1e6:.2f} M"),
        ("Size on disk (.keras, KB)", lambda m: f"{m['size_on_disk_kb']:.1f}"),
        ("float32 weights (KB)", lambda m: f"{m['float32_weights_kb']:.1f}"),
        ("Median time / epoch (s)", lambda m: f"{m['median_s_per_epoch']:.1f}"),
        ("Best val accuracy (epoch)", lambda m: f"{m['best_val_accuracy']:.4f} ({m['best_val_epoch']})"),
        ("Test accuracy", lambda m: f"{m['test_accuracy']:.4f}"),
        ("Macro precision", lambda m: f"{m['macro_precision']:.4f}"),
        ("Macro recall", lambda m: f"{m['macro_recall']:.4f}"),
    ]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    csv = [",".join(header)]
    for label, fn in lines:
        vals = [fn(m) for m in all_metrics]
        md.append("| " + " | ".join([label] + vals) + " |")
        csv.append(",".join([label] + [v.replace(",", "") for v in vals]))
    (OUT_DIR / "q4_comparison.md").write_text("\n".join(md) + "\n")
    (OUT_DIR / "q4_comparison.csv").write_text("\n".join(csv) + "\n")
    print("\n" + "\n".join(md))

    pc = ["| Class | " + " | ".join(f"{m['model']} precision | {m['model']} recall"
                                     for m in all_metrics) + " |",
          "|" + "---|" * (1 + 2 * len(all_metrics))]
    for n in names:
        cells = []
        for m in all_metrics:
            cells += [f"{m['per_class'][n]['precision']:.3f}", f"{m['per_class'][n]['recall']:.3f}"]
        pc.append(f"| {n} | " + " | ".join(cells) + " |")
    (OUT_DIR / "q4_per_class.md").write_text("\n".join(pc) + "\n")
    print("\n" + "\n".join(pc))


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=["A", "B"], default=["A", "B"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-lr-schedule", action="store_true",
                    help="constant lr = 1e-3 (the first Q4 run)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    OUT_DIR = args.out_dir

    names = class_names()
    if not args.eval_only:
        for m in args.models:
            train(m, args.epochs, augment=not args.no_augment,
                  lr_schedule=not args.no_lr_schedule)

    all_metrics = []
    for m in ["A", "B"]:                      # table always shows every trained model
        if (OUT_DIR / f"model_{m}.keras").exists():
            all_metrics.append(evaluate(m, names))
    write_tables(all_metrics, names)


if __name__ == "__main__":
    main()
