#!/usr/bin/env python3
"""
Load model and dataset, print metrics, save evaluation plots.

Usage (from repo root):
    python scripts/evaluate_model.py --model models/rf_missile_range.joblib --data data/missile_dataset.csv
If data file missing, it will attempt to generate a new dataset (2000 samples).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# If dataset not available, use same simulate_range as generator
def simulate_range(v0, theta_deg, m=1.0, A=0.01, Cd=0.5, rho=1.225, dt=0.01, max_t=200.0):
    theta = np.deg2rad(theta_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0.0, 0.0
    t = 0.0
    while t < max_t and y >= 0:
        v = np.hypot(vx, vy)
        if v > 0:
            Fd = 0.5 * rho * Cd * A * v**2
            ax = -(Fd / m) * (vx / v)
            ay = -9.80665 - (Fd / m) * (vy / v)
        else:
            ax, ay = 0.0, -9.80665
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
    return x

def ensure_dataset(path, fallback_n=2000, seed=99):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        print(f"Dataset not found at {path}. Generating fallback dataset ({fallback_n} samples).")
        rng = np.random.default_rng(seed)
        v0 = rng.uniform(50, 500, fallback_n)
        theta = rng.uniform(5, 80, fallback_n)
        m = rng.uniform(5, 200, fallback_n)
        A = rng.uniform(0.005, 0.5, fallback_n)
        Cd = rng.uniform(0.1, 1.2, fallback_n)
        rho = rng.uniform(1.0, 1.3, fallback_n)
        rows = []
        for i in range(fallback_n):
            r = simulate_range(v0[i], theta[i], m[i], A[i], Cd[i], rho[i])
            rows.append([v0[i], theta[i], m[i], A[i], Cd[i], rho[i], r])
        df = pd.DataFrame(rows, columns=['v0','theta','m','A','Cd','rho','range'])
        return df

def evaluate(model_path, data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    # load dataset
    df = ensure_dataset(data_path)
    X = df[['v0','theta','m','A','Cd','rho']].values
    y_true = df['range'].values

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print("Evaluation on dataset:")
    print(f"Samples: {len(y_true)}")
    print(f"MAE: {mae:.3f} m")
    print(f"RMSE: {rmse:.3f} m")
    print(f"R2: {r2:.4f}")

    # scatter plot: true vs pred
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    mmax = max(y_true.max(), y_pred.max())
    plt.plot([0, mmax], [0, mmax], 'r--', linewidth=1)
    plt.xlabel("True range (m)")
    plt.ylabel("Predicted range (m)")
    plt.title("Predicted vs True Range")
    plt.grid(True)
    scatter_path = os.path.join(out_dir, "pred_vs_true.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {scatter_path}")

    # residual histogram
    residuals = y_true - y_pred
    plt.figure(figsize=(6,3.5))
    plt.hist(residuals, bins=60, alpha=0.8)
    plt.xlabel("Residual (true - pred) [m]")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "residuals_hist.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved residuals histogram: {hist_path}")

    # feature importances (if available)
    fi_path = None
    try:
        importances = model.feature_importances_
        names = ['v0','theta','m','A','Cd','rho']
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(6,3.5))
        plt.bar([names[i] for i in order], importances[order])
        plt.ylabel("Importance")
        plt.title("Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(out_dir, "feature_importances.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"Saved feature importances: {fi_path}")
    except Exception:
        print("Model has no feature_importances_ attribute or plotting failed.")

    # save numeric report
    report_txt = os.path.join(out_dir, "evaluation_report.txt")
    with open(report_txt, "w") as f:
        f.write("Evaluation report\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Samples: {len(y_true)}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"R2: {r2:.6f}\n")
    print(f"Saved numeric report: {report_txt}")

def parse_args():
    p = argparse.ArgumentParser(prog="evaluate_model.py")
    p.add_argument("--model", type=str, default=None, help="Model path (joblib)")
    p.add_argument("--data", type=str, default=None, help="Dataset CSV path")
    p.add_argument("--out", type=str, default=None, help="Output directory for evaluation (plots/reports)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    model_path = args.model if args.model else os.path.join(repo_root, "models", "rf_missile_range.joblib")
    data_path = args.data if args.data else os.path.join(repo_root, "data", "missile_dataset.csv")
    out_dir = args.out if args.out else os.path.join(repo_root, "assets", "evaluation")

    evaluate(model_path, data_path, out_dir)
