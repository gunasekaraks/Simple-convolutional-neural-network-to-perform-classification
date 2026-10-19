"""
EN3150 Assignment 03 - Q2: Custom architecture design.

Model A: standard CNN (Conv2D + MaxPooling).
Model B: lightweight CNN using depthwise separable convolutions (<= 100k params).

Both models share the SAME macro-structure (4 blocks, 32->64->128->256 channels,
same classifier head) so that the only difference is standard vs depthwise
separable convolution. This makes the Q4 comparison a fair one.

Input: 64x64x3 float32 in [0, 1] (from prepare_data.load_datasets()).
Output: 5-class softmax.

Run `python models.py` to print both summaries, the hand parameter count,
and the MAC count, and to check that the hand count matches Keras.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (64, 64, 3)
NUM_CLASSES = 5
FILTERS = (32, 64, 128, 256)   # channels per block
DENSE_UNITS = 128
DROPOUT = 0.3
PARAM_LIMIT_B = 100_000


# --------------------------------------------------------------------------
# Model A: standard CNN
# Block = Conv2D 3x3 (no bias) -> BatchNorm -> ReLU -> MaxPool 2x2
# --------------------------------------------------------------------------
def build_model_a(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs
    for i, f in enumerate(FILTERS, start=1):
        # use_bias=False: BatchNorm's beta already provides a per-channel
        # offset, so a conv bias would be redundant (it gets subtracted away
        # by BN's mean-centering).
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="Model_A_standard")


# --------------------------------------------------------------------------
# Model B: depthwise separable CNN
# Block = DepthwiseConv 3x3 -> BN -> ReLU6 -> Pointwise Conv 1x1 -> BN -> ReLU6
#         -> MaxPool 2x2
# Depthwise and pointwise are written as two separate layers (instead of
# keras SeparableConv2D) so each appears in the summary and can be hand-counted.
# --------------------------------------------------------------------------
def build_model_b(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs
    for i, f in enumerate(FILTERS, start=1):
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False, name=f"dw{i}")(x)
        x = layers.BatchNormalization(name=f"bn_dw{i}")(x)
        x = layers.ReLU(max_value=6.0, name=f"relu6_dw{i}")(x)
        x = layers.Conv2D(f, 1, use_bias=False, name=f"pw{i}")(x)
        x = layers.BatchNormalization(name=f"bn_pw{i}")(x)
        x = layers.ReLU(max_value=6.0, name=f"relu6_pw{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="Model_B_depthwise_separable")


# --------------------------------------------------------------------------
# Hand calculation (formula-based, independent of Keras)
# --------------------------------------------------------------------------
def hand_count(kind):
    """Return rows of (layer, formula, trainable, non_trainable, MACs)."""
    rows = []
    c_in, hw = INPUT_SHAPE[2], INPUT_SHAPE[0]   # channels, spatial size
    k = 3
    for i, f in enumerate(FILTERS, start=1):
        if kind == "A":
            p = k * k * c_in * f
            rows.append((f"conv{i}", f"{k}x{k}x{c_in}x{f}", p, 0, hw * hw * p))
            rows.append((f"bn{i}", f"2x{f} (+2x{f} non-train.)", 2 * f, 2 * f, 0))
        else:
            p_dw = k * k * c_in
            rows.append((f"dw{i}", f"{k}x{k}x{c_in}", p_dw, 0, hw * hw * p_dw))
            rows.append((f"bn_dw{i}", f"2x{c_in} (+2x{c_in} non-train.)", 2 * c_in, 2 * c_in, 0))
            p_pw = c_in * f
            rows.append((f"pw{i}", f"1x1x{c_in}x{f}", p_pw, 0, hw * hw * p_pw))
            rows.append((f"bn_pw{i}", f"2x{f} (+2x{f} non-train.)", 2 * f, 2 * f, 0))
        c_in, hw = f, hw // 2
    p = c_in * DENSE_UNITS + DENSE_UNITS
    rows.append(("fc", f"{c_in}x{DENSE_UNITS} + {DENSE_UNITS}", p, 0, c_in * DENSE_UNITS))
    p = DENSE_UNITS * NUM_CLASSES + NUM_CLASSES
    rows.append(("output", f"{DENSE_UNITS}x{NUM_CLASSES} + {NUM_CLASSES}", p, 0, DENSE_UNITS * NUM_CLASSES))
    return rows


def keras_counts(model):
    tr = sum(int(tf.size(w)) for w in model.trainable_weights)
    ntr = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    return tr, ntr


def report(model, kind):
    model.summary()
    rows = hand_count(kind)
    print(f"\nHand calculation - Model {kind}")
    print(f"{'layer':<10}{'formula':<30}{'trainable':>11}{'non-train':>11}{'MACs':>14}")
    for name, formula, tr, ntr, macs in rows:
        print(f"{name:<10}{formula:<30}{tr:>11,}{ntr:>11,}{macs:>14,}")
    h_tr = sum(r[2] for r in rows)
    h_ntr = sum(r[3] for r in rows)
    h_macs = sum(r[4] for r in rows)
    print(f"{'TOTAL':<40}{h_tr:>11,}{h_ntr:>11,}{h_macs:>14,}")

    k_tr, k_ntr = keras_counts(model)
    assert (h_tr, h_ntr) == (k_tr, k_ntr), f"Mismatch: hand {h_tr}/{h_ntr} vs keras {k_tr}/{k_ntr}"
    print(f"Keras check: trainable {k_tr:,}, non-trainable {k_ntr:,}  -> MATCH")
    print(f"float32 weight size ~ {(k_tr + k_ntr) * 4 / 1024:.1f} KB\n")
    return h_tr, h_macs


if __name__ == "__main__":
    a_params, a_macs = report(build_model_a(), "A")
    b_params, b_macs = report(build_model_b(), "B")
    assert b_params <= PARAM_LIMIT_B, f"Model B has {b_params:,} > {PARAM_LIMIT_B:,} params"
    print(f"Model B within limit: {b_params:,} <= {PARAM_LIMIT_B:,}")
    print(f"Param reduction A/B: {a_params / b_params:.2f}x")
    print(f"MAC reduction  A/B: {a_macs / b_macs:.2f}x")
