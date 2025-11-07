import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def simulate_range(v0, theta_deg, m=1.0, A=0.01, Cd=0.5, rho=1.225, dt=0.01, max_t=200.0):
    theta = np.deg2rad(theta_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0.0, 0.0
    t = 0.0
    while t < max_t and (t == 0 or y >= 0.0):
        v = np.hypot(vx, vy)
        if v > 1e-8:
            Fd = 0.5 * rho * Cd * A * v**2
            ax = - (Fd / m) * (vx / v)
            ay = -9.80665 - (Fd / m) * (vy / v)
        else:
            ax, ay = 0.0, -9.80665
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        if y < -1000:
            break
    return max(0.0, x)

def generate_dataset(N=3000, seed=42):
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(50, 500, N)
    theta = rng.uniform(5, 80, N)
    m = rng.uniform(5, 200, N)
    A = rng.uniform(0.005, 0.5, N)
    Cd = rng.uniform(0.1, 1.2, N)
    rho = rng.uniform(1.0, 1.3, N)
    data = []
    for i in range(N):
        r = simulate_range(v0[i], theta[i], m[i], A[i], Cd[i], rho[i])
        data.append([v0[i], theta[i], m[i], A[i], Cd[i], rho[i], r])
    df = pd.DataFrame(data, columns=['v0','theta','m','A','Cd','rho','range'])
    return df

def train_model(df):
    X = df[['v0','theta','m','A','Cd','rho']]
    y = df['range']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Absolute Error:", round(mae, 2))
    print("R² Score:", round(r2, 3))
    return model, X_test, y_test, y_pred

def plot_results(y_test, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, s=8, alpha=0.7)
    maxv = max(y_test.max(), y_pred.max())
    plt.plot([0, maxv], [0, maxv], 'r--')
    plt.xlabel('True Range (m)')
    plt.ylabel('Predicted Range (m)')
    plt.title('Predicted vs True Impact Range')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = generate_dataset()
    print(df.head())
    model, X_test, y_test, y_pred = train_model(df)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_missile_range.joblib')
    print("Model saved to models/rf_missile_range.joblib")
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
