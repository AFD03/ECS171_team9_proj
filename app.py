import streamlit as st
import pandas as pd
import joblib
import math

MODEL_PATH = "model/model.joblib"

st.set_page_config(page_title="Heart Disease Risk Demo", page_icon="❤️", layout="centered")

@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["selected_features"]

def risk_bucket(p: float):
    """
    Map probability p in [0,1] to:
      - interval label like '0.20 - 0.30'
      - text like 'moderately safe' ... 'highly risk'
    """
    # Clamp
    p = max(0.0, min(1.0, float(p)))

    # Bin index 0..9
    idx = min(int(p * 10), 9)

    low = idx / 10
    high = (idx + 1) / 10

    labels = [
        "very safe",
        "safe",
        "mostly safe",
        "slightly safe",
        "borderline",
        "some risk",
        "moderate risk",
        "high risk",
        "very high risk",
        "highly risk",
    ]
    return f"{low:0.2f} - {high:0.2f}", labels[idx]

st.title("❤️ Heart Disease Risk Prediction (Demo)")
st.caption("Educational demo only — not medical advice or a diagnosis.")

# Load model
try:
    pipeline, selected_features = load_artifact()
except Exception as e:
    st.error(f"Could not load model artifact at `{MODEL_PATH}`.")
    st.code(str(e))
    st.stop()

with st.expander("What do these factors mean?"):
    st.markdown(
        """
**Age**: patient age in years.  
**Sex**: Female/Male (converted to 0/1 internally).  

**Chest pain type (cp)** (UCI-style):
- 1 = Typical angina
- 2 = Atypical angina
- 3 = Non-anginal pain
- 4 = Asymptomatic  
*(Your model uses `cp_4`, i.e., whether chest pain type is 4.)*

**Max heart rate achieved (thalach)**: highest heart rate during exercise test.

**Oldpeak**: ST depression induced by exercise relative to rest. Higher often indicates more abnormality.

**Exercise-induced angina (exang)**: 0 = No, 1 = Yes.

**Number of major vessels (ca)**: integer 0–4 vessels colored by fluoroscopy.

**Thal (thal)** (common UCI-style):
- 3 = Normal
- 6 = Fixed defect
- 7 = Reversible defect  
*(Your model uses `thal_7.0`, i.e., whether thal is 7.)*
        """
    )

# Optional: show your risk scale legend
with st.expander("Risk scale legend (probability → label)"):
    rows = []
    for i, txt in enumerate([
        "very safe","safe","mostly safe","slightly safe","borderline",
        "some risk","moderate risk","high risk","very high risk","highly risk"
    ]):
        rows.append((f"{i/10:0.2f} - {(i+1)/10:0.2f}", txt))
    st.table(pd.DataFrame(rows, columns=["Probability range", "Risk label"]))

with st.form("patient_form"):
    st.subheader("Enter patient info")

    age = st.selectbox("Age (years)", list(range(1, 121)), index=44)  # default 45

    sex_label = st.selectbox("Sex", ["Female", "Male"])

    # cp selectable with meaning
    cp_map = {
        "1 — Typical angina": 1,
        "2 — Atypical angina": 2,
        "3 — Non-anginal pain": 3,
        "4 — Asymptomatic": 4,
    }
    cp_choice = st.selectbox("Chest pain type (cp)", list(cp_map.keys()))
    cp = cp_map[cp_choice]

    thalach = st.selectbox("Max heart rate achieved (thalach)", list(range(60, 221)), index=90)  # default ~150

    # You asked: selectable whole numbers with maximum
    oldpeak = st.selectbox("Oldpeak (integer for demo)", list(range(0, 11)), index=1)

    exang_label = st.selectbox("Exercise-induced angina (exang)", ["0 — No", "1 — Yes"])
    exang = 1 if exang_label.startswith("1") else 0

    ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4], index=0)

    thal_map = {
        "3 — Normal": 3,
        "6 — Fixed defect": 6,
        "7 — Reversible defect": 7,
    }
    thal_choice = st.selectbox("Thal (thal)", list(thal_map.keys()))
    thal = thal_map[thal_choice]

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert to the EXACT feature vector your trained model expects
    sex = 1 if sex_label == "Male" else 0
    cp_4 = 1 if cp == 4 else 0
    thal_7 = 1 if thal == 7 else 0

    X = pd.DataFrame([{
        "cp_4": cp_4,
        "ca": float(ca),
        "thal_7.0": thal_7,
        "exang": int(exang),
        "oldpeak": float(oldpeak),
        "thalach": float(thalach),
        "sex": int(sex),
        "age": float(age),
    }])

    # Ensure column order matches training
    X = X[selected_features]

    try:
        proba = float(pipeline.predict_proba(X)[0][1])
        pred = int(proba >= 0.5)

        bucket_range, bucket_label = risk_bucket(proba)

        st.divider()
        st.subheader("Result")

        st.write(f"Predicted probability of heart disease: **{proba:.2f}**")
        st.write(f"Risk scale: **{bucket_range} → {bucket_label}**")

        st.progress(max(0.0, min(1.0, proba)))

        if pred == 1:
            st.error("Model prediction: **Heart Disease (higher risk)**")
        else:
            st.success("Model prediction: **No Heart Disease (lower risk)**")

        st.caption("Threshold = 0.50 for demo. Educational use only.")
    except Exception as e:
        st.error("Prediction failed (feature mismatch or model artifact issue).")
        st.code(str(e))