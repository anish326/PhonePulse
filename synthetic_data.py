"""Synthetic per-device telemetry for prototyping the anomaly model.

NOTE: synthetic data only. Replace with real telemetry collected from the app
before making any accuracy claims.
"""
import numpy as np

FEATURES = [
    "idle_temp_c",            # battery temp while idle
    "charge_temp_rise_c",     # temp rise over 10 min of charging
    "idle_drain_pct_hr",      # battery drain while idle
    "voltage_sag_mv",         # voltage drop under CPU load
    "charge_rate_ma",         # average charging current
    "accel_noise_std",        # accelerometer noise (phone stationary)
    "gyro_bias",              # gyro bias (phone stationary)
    "thermal_headroom",       # thermal headroom (higher = cooler)
]
N_FEATURES = len(FEATURES)

# Population-level mean and spread of a "healthy" phone
POP_MEAN = np.array([30.0, 3.0, 1.0, 120.0, 2200.0, 0.02, 0.004, 0.75])
POP_STD = np.array([3.0, 0.8, 0.3, 25.0, 300.0, 0.005, 0.001, 0.08])
# Window-to-window noise within one device (smaller than between devices)
WIN_STD = POP_STD * 0.25


def make_device(rng, n_windows, anomaly=None, anomaly_start=None):
    """Return (n_windows, N_FEATURES) raw windows for one device.

    anomaly: None | 'impact_damage' | 'battery_stress' | 'charge_fault'
    Windows from anomaly_start onward are anomalous.
    """
    baseline = POP_MEAN + rng.normal(0, 1, N_FEATURES) * POP_STD
    x = baseline + rng.normal(0, 1, (n_windows, N_FEATURES)) * WIN_STD
    labels = np.zeros(n_windows, dtype=int)
    if anomaly is not None:
        s = anomaly_start if anomaly_start is not None else n_windows // 2
        labels[s:] = 1
        k = n_windows - s
        if anomaly == "impact_damage":
            x[s:, 5] += rng.uniform(2, 5) * WIN_STD[5] * 3   # noisier accel
            x[s:, 6] += rng.uniform(2, 5) * WIN_STD[6] * 3   # gyro bias shift
            x[s:, 2] += rng.uniform(1, 3) * WIN_STD[2] * 2   # extra drain
        elif anomaly == "battery_stress":
            x[s:, 0] += rng.uniform(3, 6)                    # hotter idle
            x[s:, 1] += rng.uniform(1.5, 3)                  # faster heat rise
            x[s:, 3] += rng.uniform(40, 80)                  # bigger sag
            x[s:, 2] += rng.uniform(0.4, 1.0)                # more drain
            x[s:, 7] -= rng.uniform(0.1, 0.25)               # less headroom
        elif anomaly == "charge_fault":
            x[s:, 4] -= rng.uniform(500, 1000)               # low current
            x[s:, 4] += rng.normal(0, 200, k)                # erratic
            x[s:, 1] += rng.uniform(0.5, 1.5)
    return x, labels, baseline


def device_normalize(x, n_baseline=40):
    """Per-device z-score using the device's own first n_baseline windows.

    On-device this is the calibration step: the app records the first
    n_baseline windows, stores mean/std, and feeds z-scores to the model.
    """
    ref = x[:n_baseline]
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-6
    return (x - mu) / sd, mu, sd


def build_dataset(seed=0, n_devices=400, n_windows=200, n_baseline=40):
    """Train on healthy devices; return anomaly test set separately."""
    rng = np.random.default_rng(seed)
    train = []
    for _ in range(n_devices):
        x, _, _ = make_device(rng, n_windows)
        z, _, _ = device_normalize(x, n_baseline)
        train.append(z[n_baseline:])          # skip calibration windows
    x_train = np.concatenate(train).astype("float32")

    test_x, test_y = [], []
    kinds = [None, "impact_damage", "battery_stress", "charge_fault"]
    for i in range(200):
        kind = kinds[i % len(kinds)]
        x, y, _ = make_device(rng, n_windows, kind, anomaly_start=120)
        z, _, _ = device_normalize(x, n_baseline)
        test_x.append(z[n_baseline:])
        test_y.append(y[n_baseline:])
    return x_train, np.concatenate(test_x).astype("float32"), np.concatenate(test_y)
