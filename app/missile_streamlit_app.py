import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st

APP_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from missile_ml_simple import simulate_range

st.set_page_config(page_title="Missile Trajectory Predictor", layout="centered")

st.title("Missile Trajectory Predictor")
st.write("This app predicts missile impact range using Physics and Machine Learning.")

try:
    model = joblib.load(os.path.join(REPO_ROOT, "models", "rf_missile_range.joblib"))
except Exception:
    st.error("Model not found. Please train it locally using 'python scripts/missile_ml_simple.py'.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    v0 = st.slider("Initial Speed (m/s)", 50.0, 500.0, 150.0)
    theta = st.slider("Launch Angle (°)", 5.0, 80.0, 45.0)
    m = st.slider("Mass (kg)", 5.0, 200.0, 50.0)

with col2:
    A = st.slider("Cross Section Area (m²)", 0.005, 0.5, 0.02)
    Cd = st.slider("Drag Coefficient", 0.1, 1.2, 0.5)
    rho = st.slider("Air Density (kg/m³)", 1.0, 1.3, 1.225)

if st.button("Predict"):
    X = np.array([[v0, theta, m, A, Cd, rho]])
    pred = model.predict(X)[0]
    st.metric("Predicted Impact Range (m)", f"{pred:.2f}")

    dt = 0.01
    vx = v0 * np.cos(np.deg2rad(theta))
    vy = v0 * np.sin(np.deg2rad(theta))
    x, y = 0.0, 0.0
    xs, ys = [x], [y]

    while y >= 0:
        v = np.hypot(vx, vy)
        if v > 0:
            Fd = 0.5 * rho * Cd * A * v**2
            ax = -(Fd / m) * (vx / v)
            ay = -9.81 - (Fd / m) * (vy / v)
        else:
            ax, ay = 0.0, -9.81
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        xs.append(x)
        ys.append(y)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, label="Trajectory")
    ax.axvline(pred, color="r", linestyle="--", label="Predicted Range")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Altitude (m)")
    ax.legend()
    st.pyplot(fig)
