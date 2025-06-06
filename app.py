import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import gradio as gr

# Suppress any warnings to keep outputs clean
import warnings
warnings.filterwarnings("ignore")

# 1. Load and prepare data
df = pd.read_csv("sleep_deprivation_dataset_detailed.csv")

# 2. Create binary target: 0 = poor sleep (1–10), 1 = good sleep (11–20)
df['sleep_quality_binary'] = (df['Sleep_Quality_Score'] >= 11).astype(int)

# 3. Select features and target
features = [
    'Stress_Level',
    'Stroop_Task_Reaction_Time',
    'Emotion_Regulation_Score',
    'Caffeine_Intake',
    'Physical_Activity_Level',
    'Sleep_Hours'
]
X = df[features]
y = df['sleep_quality_binary']

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train logistic regression on all data
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_scaled, y)

# 6. Define the Gradio prediction function
def predict_sleep_quality(
    stress_level, stroop_time, emotion_score,
    caffeine_intake, physical_activity, sleep_hours
):
    """
    Build a row from inputs, scale with the same scaler,
    and return logistic prediction + confidence.
    """
    row = pd.DataFrame([{
        'Stress_Level': stress_level,
        'Stroop_Task_Reaction_Time': stroop_time,
        'Emotion_Regulation_Score': emotion_score,
        'Caffeine_Intake': caffeine_intake,
        'Physical_Activity_Level': physical_activity,
        'Sleep_Hours': sleep_hours
    }])
    scaled = scaler.transform(row)
    pred = log_model.predict(scaled)[0]
    prob = log_model.predict_proba(scaled)[0][pred]
    label = "Good Sleep" if pred == 1 else "Poor Sleep"
    return f"{label} (Confidence: {prob:.2f})"

# 7. Create and launch Gradio interface
demo = gr.Interface(
    fn=predict_sleep_quality,
    inputs=[
        gr.Slider(0, 10, value=5, label="Stress Level"),
        gr.Slider(0.2, 2.5, value=1.5, step=0.01, label="Stroop Task Reaction Time (sec)"),
        gr.Slider(0, 10, value=5, label="Emotion Regulation Score"),
        gr.Slider(0, 5, value=2, label="Caffeine Intake (cups)"),
        gr.Slider(0, 10, value=5, label="Physical Activity Level"),
        gr.Slider(0, 12, value=7, step=0.5, label="Sleep Hours")
    ],
    outputs="text",
    title="Sleep Quality Prediction",
    description="Move the sliders to predict whether sleep will be Good or Poor."
)

if __name__ == "__main__":
    demo.launch()
