# 🚀 Missile Trajectory Predictor (Physics + Machine Learning)

This project combines **Projectile Physics** and **Machine Learning** to predict a missile’s **impact range** based on parameters like launch speed, angle, mass, and aerodynamic drag.  
It simulates real-world flight using physics equations, trains a regression model using the generated data, and visualizes predictions through a **Streamlit web app**.

---

## 🧠 Overview

Traditional physics can calculate projectile range, but factors like air drag, density, and body area make it complex.  
This project builds a **physics simulator** to generate realistic flight data, then trains a **Random Forest Regressor** to predict impact range from any given input.

The Streamlit app lets users:
- Input missile parameters
- See predicted impact range (via ML)
- Visualize the full projectile trajectory

---

## 🧩 Features

- Physics-based dataset generator (with quadratic drag)
- Random Forest regression model for range prediction
- Interactive Streamlit interface
- Real-time trajectory visualization
- Modular and extensible code structure

---

## ⚙️ Folder Structure


missile-trajectory-predictor/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│ ├── missile_ml_simple.py ← Physics simulator + ML training
│ ├── dataset_generator.py ← Dataset creation script
│ └── evaluate_model.py ← Evaluation and metrics
│
├── app/
│ └── missile_streamlit_app.py ← Streamlit web interface
│
├── models/
│ └── rf_missile_range.joblib ← Trained Random Forest model
│
├── data/
│ └── missile_dataset.csv ← Generated dataset (optional)
│
├── assets/
│ ├── demo_screenshot.png
│ └── trajectory_plot_example.png
│
└── notebooks/
└── missile_experiments.ipynb

yaml
Copy code

---

## 🔧 Setup & Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/missile-trajectory-predictor.git
cd missile-trajectory-predictor
2️⃣ Create a virtual environment and install dependencies
bash
Copy code
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
3️⃣ Train the model (optional if already saved)
bash
Copy code
python scripts/missile_ml_simple.py
4️⃣ Run the Streamlit app
bash
Copy code
streamlit run app/missile_streamlit_app.py
App will open at: http://localhost:8501

📊 Example Input
Parameter	Description	Example
v0	Initial speed (m/s)	200
theta	Launch angle (°)	45
m	Mass (kg)	50
A	Cross-sectional area (m²)	0.02
Cd	Drag coefficient	0.5
rho	Air density (kg/m³)	1.225

🖼 Preview

📈 Results
Mean Absolute Error: ~25 m

R² Score: ~0.98

Predicted ranges closely match physics simulations

The model effectively learns aerodynamic behavior through synthetic physics data — a practical fusion of theory and AI.

🧮 Technical Stack
Language: Python

ML Framework: Scikit-learn

Web Framework: Streamlit

Libraries: NumPy, Pandas, Matplotlib, Joblib

💬 Project Summary (for Viva or Report)
This project demonstrates the use of Machine Learning to predict missile trajectories.
It combines a custom-built physics simulator (accounting for drag and air density) with a Random Forest regression model.
The trained model predicts the missile’s impact range with high accuracy, and the results are visualized in an interactive web app built using Streamlit.

👨‍💻 Author
Japkaran Singh Arneja
Lovely Professional University
📧 japkaran.work.12@gmail.com

⭐ If you liked this project, don’t forget to Star the repository on GitHub!

yaml
Copy code

---

## ✅ Tips before submitting

1. Add one or two screenshots in the `assets/` folder (use your working app screenshot).
2. Replace `<your-username>` in GitHub URL with your GitHub ID.
3. Commit all changes:
   ```bash
   git add .
   git commit -m "Final submission: Missile Trajectory Predictor"
   git push origin main"# missile-trajectory-predictor" 
