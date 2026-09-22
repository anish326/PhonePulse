"""Train the per-device anomaly autoencoder and export TFLite models.

Usage:  python train_autoencoder.py
Output: model.tflite (float32), model_int8.tflite (NPU-friendly), model_meta.json
"""
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from synthetic_data import build_dataset, FEATURES, N_FEATURES

SEED = 0
tf.keras.utils.set_random_seed(SEED)

x_train, x_test, y_test = build_dataset(seed=SEED)
split = int(0.9 * len(x_train))
x_tr, x_val = x_train[:split], x_train[split:]


def build_model():
    inp = tf.keras.Input(shape=(N_FEATURES,), name="features")
    h = tf.keras.layers.Dense(16, activation="relu")(inp)
    z = tf.keras.layers.Dense(4, activation="relu")(h)
    h = tf.keras.layers.Dense(16, activation="relu")(z)
    out = tf.keras.layers.Dense(N_FEATURES, name="recon")(h)
    return tf.keras.Model(inp, out)


model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
model.fit(
    x_tr, x_tr,
    validation_data=(x_val, x_val),
    epochs=60, batch_size=256, verbose=2,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=6, restore_best_weights=True)],
)


def recon_error(m, x):
    return np.mean((m.predict(x, verbose=0) - x) ** 2, axis=1)


# Alert threshold = 99th percentile of error on healthy validation data
threshold = float(np.percentile(recon_error(model, x_val), 99))
err_test = recon_error(model, x_test)
auc = roc_auc_score(y_test, err_test)
pred = err_test > threshold
tpr = float(pred[y_test == 1].mean())
fpr = float(pred[y_test == 0].mean())
print(f"AUROC={auc:.3f}  TPR={tpr:.3f}  FPR={fpr:.3f}  threshold={threshold:.4f}")

# ---- Export float32 TFLite
conv = tf.lite.TFLiteConverter.from_keras_model(model)
open("model.tflite", "wb").write(conv.convert())


# ---- Export int8 TFLite (better fit for NPU/DSP delegates)
def rep_data():
    for i in range(0, 1000):
        yield [x_train[i:i + 1]]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
open("model_int8.tflite", "wb").write(conv.convert())

# ---- Metadata the Android app needs
json.dump({
    "features": FEATURES,
    "threshold_mse": threshold,
    "calibration_windows": 40,
    "normalization": "per-device z-score using first 40 windows",
    "data": "synthetic (prototype); replace with real telemetry",
    "metrics_synthetic": {"auroc": auc, "tpr": tpr, "fpr": fpr},
}, open("model_meta.json", "w"), indent=2)
print("Saved model.tflite, model_int8.tflite, model_meta.json")
