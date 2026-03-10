import streamlit as st
import sys
import os
import pandas as pd
from PIL import Image
import numpy as np

# Add project root to path so we can import backend logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference.manager import JerryScanModelManager
from backend.inference.core import JerryScanAnomalibModel
from model_lab.data_loader import LabDataLoader
from model_lab.metrics_calculator import MetricsCalculator

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

# Initialize Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "metrics1" not in st.session_state:
    st.session_state.metrics1 = None
if "metrics2" not in st.session_state:
    st.session_state.metrics2 = None
if "config" not in st.session_state:
    st.session_state.config = {}

# Clear results if configuration changes
current_config = {
    "model1": selected_model_set,
    "model2": second_model_set,
    "angle": selected_angle,
    "dataset": dataset_path
}
if st.session_state.config != current_config:
    st.session_state.results = None
    st.session_state.metrics1 = None
    st.session_state.metrics2 = None
    st.session_state.config = current_config

if st.button("🚀 Run Full Benchmark"):
    st.write(f"**Status:** Evaluating {selected_model_set} ({selected_angle})...")
    progress = st.progress(0)
    temp_results = []
    
    model1 = manager.get_model(selected_angle, model_name=selected_model_set)
    model2 = manager.get_model(selected_angle, model_name=second_model_set) if second_model_set else None
    
    for i, sample in enumerate(samples):
        with open(sample["path"], "rb") as f:
            img_bytes = f.read()
            p1 = model1.predict(img_bytes)
            p2 = model2.predict(img_bytes) if model2 else None
            
            # Label Mapping for Correctness
            m1_status = p1["status"].lower() # "pass" or "fail"
            true_cat = sample["label"].lower() # "normal" or "fault"
            is_m1_correct = (m1_status == "pass" and true_cat == "normal") or (m1_status == "fail" and true_cat == "fault")
            
            res = {
                "Filename": sample["filename"],
                "True Label": true_cat,
                "M1 Pred": m1_status,
                "M1 Score %": p1["score_percentage"],
                "M1 Correct": is_m1_correct,
                "prediction": p1,
                "sample": sample
            }
            if p2:
                m2_status = p2["status"].lower()
                is_m2_correct = (m2_status == "pass" and true_cat == "normal") or (m2_status == "fail" and true_cat == "fault")
                res.update({
                    "M2 Pred": m2_status,
                    "M2 Score %": p2["score_percentage"],
                    "M2 Correct": is_m2_correct,
                    "prediction2": p2
                })
            temp_results.append(res)
        progress.progress((i + 1) / len(samples))
    
    # Store in session state
    st.session_state.results = temp_results
    st.session_state.metrics1 = MetricsCalculator.calculate_metrics([{"True Label": r["True Label"], "Pred Label": r["M1 Pred"], "Score %": r["M1 Score %"]} for r in temp_results])
    if second_model_set:
        st.session_state.metrics2 = MetricsCalculator.calculate_metrics([{"True Label": r["True Label"], "Pred Label": r["M2 Pred"], "Score %": r["M2 Score %"]} for r in temp_results])
    st.success("Benchmark Complete!")

# Rendering Logic (Persistent)
if st.session_state.results is not None:
    results = st.session_state.results
    metrics1 = st.session_state.metrics1
    metrics2 = st.session_state.metrics2
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["prediction", "prediction2", "sample"]} for r in results])

    st.subheader(f"📊 Performance: {selected_model_set}")
    m1_col1, m1_col2, m1_col3, m1_col4 = st.columns(4)
    m1_col1.metric("Accuracy", f"{metrics1['Accuracy']*100:.1f}%")
    m1_col2.metric("AUROC", f"{metrics1['AUROC']:.3f}")
    m1_col3.metric("F1 Score", f"{metrics1['F1 Score']:.3f}")
    m1_col4.metric("Recall", f"{metrics1['Recall']:.3f}")

    if second_model_set and metrics2:
        st.subheader(f"📊 Performance: {second_model_set}")
        m2_col1, m2_col2, m2_col3, m2_col4 = st.columns(4)
        m2_col1.metric("Accuracy", f"{metrics2['Accuracy']*100:.1f}%", delta=f"{(metrics2['Accuracy'] - metrics1['Accuracy'])*100:.1f}%")
        m2_col2.metric("AUROC", f"{metrics2['AUROC']:.3f}", delta=f"{metrics2['AUROC'] - metrics1['AUROC']:.3f}")
        m2_col3.metric("F1 Score", f"{metrics2['F1 Score']:.3f}", delta=f"{metrics2['F1 Score'] - metrics1['F1 Score']:.3f}")
        m2_col4.metric("Recall", f"{metrics2['Recall']:.3f}", delta=f"{metrics2['Recall'] - metrics1['Recall']:.3f}")

    st.subheader("Detailed Results Table")
    st.dataframe(df, use_container_width=True)

    # Simplified Helper for B64 Images
    def get_image_from_b64(b64_str):
        import base64
        from io import BytesIO
        img_b64 = b64_str.split(",")[1]
        return BytesIO(base64.b64decode(img_b64))

    # Side-by-Side Analysis
    st.subheader("🔍 Sample Analysis")
    show_mode = st.radio("Show samples:", ["Errors Only", "All Samples"], horizontal=True)
    
    display_samples = results
    if show_mode == "Errors Only":
        display_samples = [r for r in results if not r["M1 Correct"] or (second_model_set and not r.get("M2 Correct", True))]

    if not display_samples:
        st.write("No samples to display in this mode.")
    else:
        for err in display_samples:
            with st.expander(f"📄 {err['Filename']} | True: {err['True Label']}", expanded=False):
                if second_model_set:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Primary: {selected_model_set}**")
                        status_color = "green" if err['M1 Correct'] else "red"
                        st.markdown(f"Status: :{status_color}[{err['M1 Pred'].upper()}] ({err['M1 Score %']:.1f}%)")
                        st.image(get_image_from_b64(err["prediction"]["heatmap_image"]), use_container_width=True)
                    with col2:
                        st.markdown(f"**Secondary: {second_model_set}**")
                        status_color = "green" if err.get('M2 Correct') else "red"
                        st.markdown(f"Status: :{status_color}[{err['M2 Pred'].upper()}] ({err['M2 Score %']:.1f}%)")
                        if "prediction2" in err:
                            st.image(get_image_from_b64(err["prediction2"]["heatmap_image"]), use_container_width=True)
                else:
                    st.write(f"Pred: {err['M1 Pred']} ({err['M1 Score %']:.1f}%)")
                    st.image(get_image_from_b64(err["prediction"]["heatmap_image"]), width=400)
