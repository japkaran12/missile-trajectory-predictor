import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scripts.missile_ml_simple import simulate_range

st.set_page_config(page_title="Missile Trajectory Predictor", layout="centered")

st.title("Missile Trajectory Predictor")
st.write("This app predicts missile impact range using Physics + Machine Learning.")

model = joblib.load('models/rf_missile_range.joblib')

v0 = st.slider("Initial Speed (m/s)", 50.0, 500.0, 150.0)
theta = st.slider("Launch Angle (°)", 5.0, 80.0, 45.0)
m = st.slider("Mass (kg)", 5.0, 200.0, 50.0)
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
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, ys, label='Trajectory')
    ax.axvline(pred, color='r', linestyle='--', label='Predicted Range')
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Altitude (m)")
    ax.legend()
    st.pyplot(fig)
