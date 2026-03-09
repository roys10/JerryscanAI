import streamlit as st
import sys
import os
import pandas as pd
from PIL import Image
import numpy as np

# Add project root to path so we can import backend logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference.manager import JerryScanModelManager
from backend.inference.core import JerryScanPadimModel
from model_lab.data_loader import LabDataLoader

st.set_page_config(page_title="Jerryscan Model Lab", layout="wide")

# Hide Streamlit "Deploy" button and extra padding
st.markdown("""
    <style>
    .stAppDeployButton {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Jerryscan Model Lab")
st.markdown("---")

# Cache the manager so we don't reload/reprint on every rerun
@st.cache_resource
def get_manager():
    mgr = JerryScanModelManager()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    mgr.load_all_models(models_dir)
    return mgr

manager = get_manager()

model_names = manager.get_model_names()
if not model_names:
    st.sidebar.error("No models found in /models directory!")
    st.stop()

selected_model_set = st.sidebar.selectbox("Select Model (Primary)", model_names)
compare_mode = st.sidebar.checkbox("Compare with another model?")
second_model_set = None
if compare_mode:
    second_model_set = st.sidebar.selectbox("Select Model (Secondary)", model_names, index=1 if len(model_names) > 1 else 0)

available_angles = list(manager.models[selected_model_set].keys())
selected_angle = st.sidebar.selectbox("Select Camera Angle", available_angles)

# Sidebar - Dataset Selection
st.sidebar.header("Dataset Configuration")
dataset_path = st.sidebar.text_input("Dataset Path (root folder)", value="./test_dataset")

if not os.path.exists(dataset_path):
    st.warning(f"Dataset path `{dataset_path}` does not exist yet. Please create it or provide a valid path.")
    st.stop()

loader = LabDataLoader(dataset_path, selected_angle)
samples = loader.get_samples()

if not samples:
    st.info(f"No samples found for angle `{selected_angle}`. Please populate `test_dataset/{selected_angle}/normal` and `/fault`.")
    st.stop()

# Main Dashboard
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(samples))
col2.metric("Normal", len([s for s in samples if s["label"] == "normal"]))
col3.metric("Fault", len([s for s in samples if s["label"] == "fault"]))

# Run Evaluation
if st.button("🚀 Run Full Benchmark"):
    st.write(f"Evaluating {selected_model_set} - {selected_angle}...")
    # Placeholder for metric calculation
    progress = st.progress(0)
    results = []
    
    model1 = manager.get_model(selected_angle, model_name=selected_model_set)
    model2 = manager.get_model(selected_angle, model_name=second_model_set) if second_model_set else None
    
    for i, sample in enumerate(samples):
        with open(sample["path"], "rb") as f:
            img_bytes = f.read()
            p1 = model1.predict(img_bytes)
            p2 = model2.predict(img_bytes) if model2 else None
            
            res = {
                "Filename": sample["filename"],
                "True Label": sample["label"],
                "M1 Pred": p1["status"].lower(),
                "M1 Score %": p1["score_percentage"],
                "M1 Correct": (p1["status"].lower() == sample["label"].lower()),
                "prediction": p1,
                "sample": sample
            }
            if p2:
                res.update({
                    "M2 Pred": p2["status"].lower(),
                    "M2 Score %": p2["score_percentage"],
                    "M2 Correct": (p2["status"].lower() == sample["label"].lower()),
                    "prediction2": p2
                })
            results.append(res)
        progress.progress((i + 1) / len(samples))
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["prediction", "prediction2", "sample"]} for r in results])
    st.success("Benchmark Complete!")
    
    # Calculate Metrics for M1
    m1_data = [{"True Label": r["True Label"], "Pred Label": r["M1 Pred"], "Score %": r["M1 Score %"]} for r in results]
    metrics1 = MetricsCalculator.calculate_metrics(m1_data)
    
    st.subheader(f"📊 Performance: {selected_model_set}")
    m1_col1, m1_col2, m1_col3, m1_col4 = st.columns(4)
    m1_col1.metric("Accuracy", f"{metrics1['Accuracy']*100:.1f}%")
    m1_col2.metric("AUROC", f"{metrics1['AUROC']:.3f}")
    m1_col3.metric("F1 Score", f"{metrics1['F1 Score']:.3f}")
    m1_col4.metric("Recall", f"{metrics1['Recall']:.3f}")

    if second_model_set:
        m2_data = [{"True Label": r["True Label"], "Pred Label": r["M2 Pred"], "Score %": r["M2 Score %"]} for r in results]
        metrics2 = MetricsCalculator.calculate_metrics(m2_data)
        
        st.subheader(f"📊 Performance: {second_model_set}")
        m2_col1, m2_col2, m2_col3, m2_col4 = st.columns(4)
        m2_col1.metric("Accuracy", f"{metrics2['Accuracy']*100:.1f}%", delta=f"{(metrics2['Accuracy'] - metrics1['Accuracy'])*100:.1f}%")
        m2_col2.metric("AUROC", f"{metrics2['AUROC']:.3f}", delta=f"{metrics2['AUROC'] - metrics1['AUROC']:.3f}")
        m2_col3.metric("F1 Score", f"{metrics2['F1 Score']:.3f}", delta=f"{metrics2['F1 Score'] - metrics1['F1 Score']:.3f}")
        m2_col4.metric("Recall", f"{metrics2['Recall']:.3f}", delta=f"{metrics2['Recall'] - metrics1['Recall']:.3f}")

    st.subheader("Detailed Results Table")
    st.dataframe(df, use_container_width=True)

    # Error Gallery
    st.subheader("🔍 Error Analysis (Primary Model)")
    errors = [r for r in results if not r["M1 Correct"]]
    if errors:
        cols = st.columns(3)
        for idx, err in enumerate(errors):
            with cols[idx % 3]:
                st.write(f"**{err['Filename']}**")
                st.write(f"True: {err['True Label']} | Pred: {err['M1 Pred']}")
                
                import base64
                from io import BytesIO
                
                img_b64 = err["prediction"]["heatmap_image"].split(",")[1]
                st.image(BytesIO(base64.b64decode(img_b64)), caption=f"M1 Heatmap ({err['M1 Score %']:.1f}%)")
    else:
        st.write("Zero errors detected for primary model!")
