"""
EN3150 Assignment 03 - Q5: fine-tuning lightweight state-of-the-art models.

Fine-tunes two ImageNet-pretrained mobile architectures on the SAME data
split as Models A and B, then evaluates them on the test set:

    mobilenetv2     MobileNetV2 (alpha = 1.0)
    efficientnetb0  EfficientNet-B0

Input handling
    The dataset is 64x64 (assignment limit). The pretrained backbones were
    trained on much larger images, so the 64x64 image is upsampled inside
    the model (bilinear) to --size (default 128). Upsampling adds no new
    information; it only puts the features at a scale the pretrained filters
    expect. The model's real input is still the 64x64 image, so the
    comparison with Models A and B is fair.
    Each backbone gets its own preprocessing, also inside the model:
        MobileNetV2     pixels scaled from [0, 255] to [-1, 1]
        EfficientNetB0  takes [0, 255] directly (it normalises internally)

Two-stage fine-tuning
    Stage 1 (feature extraction): backbone frozen, only the new 5-class head
        is trained. Adam, lr = 1e-3.
    Stage 2 (fine-tuning): the top --unfreeze layers of the backbone are
        unfrozen and trained together with the head at a much lower learning
        rate (Adam, lr = 1e-5 by default) so the pretrained features are
        adjusted gently and not destroyed.
    BatchNorm layers stay frozen and in inference mode throughout: their
    ImageNet statistics are kept, which is the standard practice when
    fine-tuning on a small dataset.

Same as Q4: augmentation on the training set only, weights from the epoch
with the best VALIDATION accuracy are kept, test set used once at the end.

Usage
    python train_sota.py                          # both models
    python train_sota.py --models mobilenetv2
    python train_sota.py --size 224               # larger upsampling (use a GPU)
    python train_sota.py --epochs1 10 --epochs2 15
    python train_sota.py --eval-only

Outputs (results/q5/)
    model_<name>.keras       best-val weights, inference-only
    history_<name>.json      curves for both stages + epoch times
    curves_<name>.png        loss/accuracy; vertical line where stage 2 starts
    confusion_<name>.png     test confusion matrix
    metrics_<name>.json      all test metrics, params, size, MACs
    q5_comparison.md/.csv    the two SOTA models side by side
    q6_comparison.md/.csv    Model B vs the two SOTA models (needs results/q4)
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from prepare_data import load_datasets
from train_eval import EpochTimer, class_names, confusion, make_augmenter

# Harmless: the best checkpoint may come from stage 1, whose optimizer had
# fewer variables than the stage 2 optimizer. Only the model weights matter.
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 5
OUT_DIR = Path("results/q5")
Q4_DIR = Path("results/q4")
COLORS = {"train": "#2a78d6", "val": "#eb6834"}

BACKBONES = {
    "mobilenetv2": ("MobileNetV2", keras.applications.MobileNetV2),
    "efficientnetb0": ("EfficientNet-B0", keras.applications.EfficientNetB0),
}


# ---------------------------------------------------------------- model
def build(name, size, weights="imagenet"):
    """Return (model, backbone). Model input: 64x64x3 float32 in [0, 255]."""
    _, ctor = BACKBONES[name]
    backbone = ctor(include_top=False, weights=weights, input_shape=(size, size, 3))

    inputs = keras.Input(shape=(64, 64, 3), name="input_64x64")
    x = layers.Resizing(size, size, interpolation="bilinear", name="upsample")(inputs)
    if name == "mobilenetv2":
        # same as keras.applications.mobilenet_v2.preprocess_input
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="to_minus1_1")(x)
    # EfficientNet: no rescaling here, the backbone normalises [0, 255] itself.

    # training=False keeps BatchNorm in inference mode even after unfreezing.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name=f"{name}_finetune"), backbone


def unfreeze_top(backbone, n):
    """Make the last n backbone layers trainable, except BatchNorm."""
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


# ---------------------------------------------------------------- counting
def _walk(layer):
    """Yield every leaf layer, descending into nested models."""
    if hasattr(layer, "layers"):
        for sub in layer.layers:
            yield from _walk(sub)
    else:
        yield layer


def count_macs(model):
    """Multiply-accumulates for one forward pass (conv, depthwise, dense).

    Conv / depthwise: H_out * W_out * (number of kernel weights)
    Dense:            N_in * N_out
    Same definition as the hand calculation used for Models A and B.
    """
    total = 0
    for layer in _walk(model):
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            h, w = layer.output.shape[1:3]
            total += int(h) * int(w) * int(np.prod(layer.kernel.shape))
        elif isinstance(layer, layers.Dense):
            total += int(np.prod(layer.kernel.shape))
    return total


def count_params(model):
    return int(sum(np.prod(w.shape) for w in model.weights))


# ---------------------------------------------------------------- data
def get_data(augment=True):
    # normalize=False -> float32 in [0, 255]; preprocessing is inside the model.
    train_ds, val_ds, test_ds = load_datasets(batch_size=BATCH_SIZE, normalize=False, seed=SEED)
    if augment:
        aug = make_augmenter()
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds, test_ds


# ---------------------------------------------------------------- training
def train(name, args):
    title = BACKBONES[name][0]
    print(f"\n=== {title}: stage 1 (backbone frozen, train head) ===")
    keras.utils.set_random_seed(SEED)
    model, backbone = build(name, args.size, args.weights)
    backbone.trainable = False

    train_ds, val_ds, _ = get_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_DIR / f"_best_{name}.weights.h5"
    timer = EpochTimer()
    # One checkpoint across both stages: keeps the best val accuracy overall.
    ckpt = keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True)

    model.compile(optimizer=keras.optimizers.Adam(args.lr1),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs1,
                   callbacks=[timer, ckpt], verbose=2)
    times1 = list(timer.times)

    print(f"\n=== {title}: stage 2 (unfreeze top {args.unfreeze} layers, lr {args.lr2}) ===")
    unfreeze_top(backbone, args.unfreeze)
    # Recompile so the change in trainable layers takes effect.
    model.compile(optimizer=keras.optimizers.Adam(args.lr2),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds,
                   initial_epoch=args.epochs1, epochs=args.epochs1 + args.epochs2,
                   callbacks=[timer, ckpt], verbose=2)

    # Restore best weights and save an inference-only copy (no optimizer state).
    model.load_weights(ckpt_path)
    deploy, _ = build(name, args.size, weights=None)
    deploy.set_weights(model.get_weights())
    deploy.save(OUT_DIR / f"model_{name}.keras")
    ckpt_path.unlink(missing_ok=True)

    h = {k: [float(v) for v in h1.history[k] + h2.history[k]] for k in h1.history}
    h.update({
        "epoch_time_s": times1 + list(timer.times),
        "stage1_epochs": args.epochs1, "stage2_epochs": args.epochs2,
        "upsample_size": args.size, "unfreeze_layers": args.unfreeze,
        "lr_stage1": args.lr1, "lr_stage2": args.lr2,
    })
    (OUT_DIR / f"history_{name}.json").write_text(json.dumps(h, indent=2))
    plot_curves(name, h)


def plot_curves(name, h):
    ep = np.arange(1, len(h["loss"]) + 1)
    best = int(np.argmax(h["val_accuracy"])) + 1
    s1 = h["stage1_epochs"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, h["loss"], color=COLORS["train"], lw=2, label="training")
    a1.plot(ep, h["val_loss"], color=COLORS["val"], lw=2, label="validation")
    a2.plot(ep, h["accuracy"], color=COLORS["train"], lw=2, label="training")
    a2.plot(ep, h["val_accuracy"], color=COLORS["val"], lw=2, label="validation")
    for ax, t in ((a1, "Loss"), (a2, "Accuracy")):
        ax.axvline(s1 + 0.5, color="#bbbbbb", lw=0.8, ls="--")
        ax.axvline(best, color="#888888", lw=1, ls=":")
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="#dddddd", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    a1.set_ylabel("Cross-entropy loss")
    a2.set_ylabel("Accuracy")
    fig.suptitle(f"{BACKBONES[name][0]}: dashed = fine-tuning starts (epoch {s1 + 1}), "
                 f"dotted = best val epoch {best}, weights kept")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"curves_{name}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- evaluation
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

    h = json.loads((OUT_DIR / f"history_{name}.json").read_text())
    s1 = h["stage1_epochs"]
    t1, t2 = h["epoch_time_s"][:s1], h["epoch_time_s"][s1:]
    total = count_params(model)
    metrics = {
        "model": BACKBONES[name][0],
        "test_accuracy": float(tp.sum() / cm.sum()),
        "per_class": {n: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "support": int(cm[i].sum())} for i, n in enumerate(names)},
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "confusion_matrix": cm.tolist(),
        "total_params": total,
        "macs": count_macs(model),
        "size_on_disk_mb": os.path.getsize(OUT_DIR / f"model_{name}.keras") / 1024 ** 2,
        "float32_weights_mb": total * 4 / 1024 ** 2,
        "upsample_size": h["upsample_size"],
        # median, skipping the first epoch of each stage (graph tracing)
        "median_s_per_epoch_stage1": float(np.median(t1[1:] or t1)),
        "median_s_per_epoch_stage2": float(np.median(t2[1:] or t2)) if t2 else None,
        "best_val_epoch": int(np.argmax(h["val_accuracy"])) + 1,
        "best_val_accuracy": float(max(h["val_accuracy"])),
        "best_val_accuracy_stage1": float(max(h["val_accuracy"][:s1])),
    }
    (OUT_DIR / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(name, cm, names)

    print(f"\n{metrics['model']} - test accuracy {metrics['test_accuracy']:.4f}")
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
    ax.set_title(f"{BACKBONES[name][0]}: test confusion matrix (acc {np.trace(cm) / cm.sum():.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def _table(header, rows, stem):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (OUT_DIR / f"{stem}.md").write_text("\n".join(md) + "\n")
    (OUT_DIR / f"{stem}.csv").write_text(
        "\n".join(",".join(c.replace(",", "") for c in r) for r in [header] + rows) + "\n")
    print("\n" + "\n".join(md))


def write_tables(all_metrics):
    if not all_metrics:
        return
    header = ["Metric"] + [m["model"] for m in all_metrics]
    spec = [
        ("Total parameters", lambda m: f"{m['total_params']:,}"),
        ("Model size on disk (MB)", lambda m: f"{m['size_on_disk_mb']:.2f}"),
        ("float32 weights (MB)", lambda m: f"{m['float32_weights_mb']:.2f}"),
        ("MACs per image", lambda m: f"{m['macs'] / 1e6:.1f} M"),
        ("Upsampled input", lambda m: f"{m['upsample_size']}x{m['upsample_size']}"),
        ("Time / epoch, head only (s)", lambda m: f"{m['median_s_per_epoch_stage1']:.1f}"),
        ("Time / epoch, fine-tuning (s)",
         lambda m: "-" if m["median_s_per_epoch_stage2"] is None else f"{m['median_s_per_epoch_stage2']:.1f}"),
        ("Best val acc, head only", lambda m: f"{m['best_val_accuracy_stage1']:.4f}"),
        ("Best val acc, overall (epoch)", lambda m: f"{m['best_val_accuracy']:.4f} ({m['best_val_epoch']})"),
        ("Test accuracy", lambda m: f"{m['test_accuracy']:.4f}"),
        ("Macro precision", lambda m: f"{m['macro_precision']:.4f}"),
        ("Macro recall", lambda m: f"{m['macro_recall']:.4f}"),
    ]
    _table(header, [[label] + [fn(m) for m in all_metrics] for label, fn in spec], "q5_comparison")

    # Q6 table: Model B (from Q4 results) against the SOTA models.
    b_path = Q4_DIR / "metrics_B.json"
    if not b_path.exists():
        print(f"\n({b_path} not found: skipping q6_comparison)")
        return
    b = json.loads(b_path.read_text())
    b_total = b["trainable_params"] + b["non_trainable_params"]
    cols = [("Model B (custom)", b_total, b["size_on_disk_kb"] / 1024, b["macs"], b["test_accuracy"])]
    cols += [(m["model"], m["total_params"], m["size_on_disk_mb"], m["macs"], m["test_accuracy"])
             for m in all_metrics]
    header = ["Metric"] + [c[0] for c in cols]
    rows = [
        ["Test accuracy"] + [f"{c[4]:.4f}" for c in cols],
        ["Total parameters"] + [f"{c[1]:,}" for c in cols],
        ["Parameters vs Model B"] + [f"{c[1] / b_total:.1f}x" for c in cols],
        ["Size on disk (MB)"] + [f"{c[2]:.2f}" for c in cols],
        ["Size vs Model B"] + [f"{c[2] / cols[0][2]:.1f}x" for c in cols],
        ["MACs per image"] + [f"{c[3] / 1e6:.1f} M" for c in cols],
        ["MACs vs Model B"] + [f"{c[3] / b['macs']:.1f}x" for c in cols],
    ]
    _table(header, rows, "q6_comparison")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(BACKBONES), default=list(BACKBONES))
    ap.add_argument("--size", type=int, default=128,
                    help="upsampled input size fed to the backbone (96/128/160/192/224)")
    ap.add_argument("--epochs1", type=int, default=10, help="stage 1 epochs (head only)")
    ap.add_argument("--epochs2", type=int, default=10, help="stage 2 epochs (fine-tuning)")
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-5)
    ap.add_argument("--unfreeze", type=int, default=40,
                    help="number of top backbone layers unfrozen in stage 2")
    ap.add_argument("--weights", default="imagenet",
                    help="'imagenet' (default) or 'none' for random init (testing only)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.weights = None if args.weights.lower() == "none" else args.weights
    OUT_DIR = args.out_dir

    if not args.eval_only:
        for name in args.models:
            train(name, args)

    names = class_names()
    all_metrics = [evaluate(n, names) for n in BACKBONES
                   if (OUT_DIR / f"model_{n}.keras").exists()]
    write_tables(all_metrics)


if __name__ == "__main__":
    main()
