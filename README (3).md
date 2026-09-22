# Anomaly model (ml/)

Small autoencoder that learns what "normal" looks like for **one phone** and flags drift.

- **Input:** 8 telemetry features per window (battery temp, charge heat rise, idle drain, voltage sag, charge rate, accelerometer noise, gyro bias, thermal headroom).
- **Per-device baseline:** the app records the first 40 windows on the device, stores their mean/std, and feeds z-scores to the model. Same model file works for any phone.
- **Output:** reconstruction error. Above `threshold_mse` (in `model_meta.json`) = "unusual behavior detected".
- **Exports:** `model.tflite` (float32) and `model_int8.tflite` (quantized, for NPU/DSP delegates via LiteRT).

## Run
```
pip install -r requirements.txt
python train_autoencoder.py
```

## Honest limitations
- Trained and evaluated on **synthetic** data (`synthetic_data.py`). Reported AUROC/TPR/FPR describe the simulator, not real-world damage detection.
- The model flags *unusual behavior*, not confirmed physical damage.
- Next step: collect real telemetry from the app (including scripted drop tests) and retrain.
