from pathlib import Path
import streamlit as st
import pandas as pd
import pefile
import joblib
import numpy as np


ROOT = Path(__file__).parent
PIPELINE_MODEL = ROOT / "samples" / "model_pipeline.joblib"   # old big dataset model
PE_MODEL = ROOT / "samples" / "model_pe.joblib"               # new PE feature model
FALLBACK_MODEL = ROOT / "samples" / "model.joblib"            # backup model
LOG_PATH = ROOT / "samples" / "scan_log.csv"

st.set_page_config(page_title="AI Malware Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI-Powered File Safety Checker")
st.write("Upload a Windows PE file (.exe or .dll) to check if it's **Safe or Malware** using a trained ML model.")


EICAR_SUB = b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE"
EICAR_FULL = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$" + EICAR_SUB + b"!$H+H*"

def extract_pe_features_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Extract key PE features for ML input."""
    try:
        pe = pefile.PE(data=file_bytes, fast_load=True)
        pe.parse_data_directories()
        sections = pe.sections or []
        entropies = [s.get_entropy() for s in sections] if sections else [0.0]

        features = {
            "NumberOfSections": len(sections),
            "ImageBase": int(getattr(pe.OPTIONAL_HEADER, "ImageBase", 0)),
            "SectionMaxEntropy": float(max(entropies)),
            "SectionMinEntropy": float(min(entropies)),
            "DllCharacteristics": int(getattr(pe.OPTIONAL_HEADER, "DllCharacteristics", 0)),
        }
        return pd.DataFrame([features])
    except Exception as e:
        raise RuntimeError(f"❌ Failed to extract PE features: {e}")

def load_first_model():
    """Try loading the PE model first, then the pipeline, then fallback."""
    for path in [PE_MODEL, PIPELINE_MODEL, FALLBACK_MODEL]:
        if path.exists():
            try:
                model = joblib.load(path)
                return model, path
            except Exception:
                continue
    return None, None

def adapt_features_for_pipeline(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to reshape PE feature dataframe for complex pipeline input."""
    try:
        pre = model.named_steps.get("pre", None) if hasattr(model, "named_steps") else None
        if pre is None or not hasattr(pre, "transformers_"):
            return features_df

        expected_cols = []
        for _, _, cols in pre.transformers_:
            if isinstance(cols, (list, tuple)):
                expected_cols.extend(cols)

        adapted = pd.DataFrame(columns=expected_cols)
        for c in expected_cols:
            adapted[c] = features_df[c].values[0] if c in features_df else 0.0
        return adapted
    except Exception as e:
        raise RuntimeError(f"Feature adaptation failed: {e}")

uploaded_file = st.file_uploader("📁 Upload an executable file", type=["exe", "dll"])

if not uploaded_file:
    st.warning("⬆️ Please upload a file to start scanning.")
    st.info("Tip: You can use a **safe EICAR test file** (rename it to `.exe`) to test detection.")
    st.stop()

file_bytes = uploaded_file.read()


contains_eicar = (EICAR_FULL in file_bytes) or (EICAR_SUB in file_bytes)
if contains_eicar:
    st.success("✅ EICAR test signature found inside the uploaded file (safe test).")


try:
    if contains_eicar:
        # Use dummy data for EICAR test
        features_df = pd.DataFrame([{
            "NumberOfSections": 0,
            "ImageBase": 0,
            "SectionMaxEntropy": 0.0,
            "SectionMinEntropy": 0.0,
            "DllCharacteristics": 0,
        }])
    else:
        st.info("⏳ Extracting PE header features...")
        features_df = extract_pe_features_from_bytes(file_bytes)
        st.success("✅ Features extracted successfully.")
except Exception as e:
    st.error(str(e))
    st.stop()

st.dataframe(features_df)


model, model_path = load_first_model()
if model is None:
    st.error(f"❌ No trained model found.\nExpected one of:\n- {PE_MODEL}\n- {PIPELINE_MODEL}\n- {FALLBACK_MODEL}")
    st.stop()

st.info(f"Loaded model from: **{model_path.name}**")

try:
    adapted_df = adapt_features_for_pipeline(model, features_df)
    preds = model.predict(adapted_df)
    pred = preds[0]

    # Compute confidence if available
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(adapted_df)
        confidence = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])

    # Normalize label output
    if isinstance(pred, (int, np.integer)):
        label = "Malware" if int(pred) == 1 else "Benign"
    else:
        label = str(pred)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.subheader("🔎 Scan Result")

if label.lower().startswith("mal"):
    st.error(f"⚠️ MALWARE DETECTED — Confidence: {confidence * 100:.2f}%" if confidence is not None else "⚠️ MALWARE DETECTED")
else:
    st.success(f"✅ SAFE FILE — Confidence: {confidence * 100:.2f}%" if confidence is not None else "✅ SAFE FILE")

if confidence is not None:
    st.progress(min(1.0, confidence if label.lower().startswith("mal") else 1.0 - confidence))

try:
    log_row = pd.DataFrame([{
        "filename": uploaded_file.name,
        "prediction": label,
        "confidence": round(float(confidence), 3) if confidence is not None else "",
        "model_used": model_path.name
    }])
    if LOG_PATH.exists():
        old_log = pd.read_csv(LOG_PATH)
        new_log = pd.concat([old_log, log_row], ignore_index=True)
    else:
        new_log = log_row
    new_log.to_csv(LOG_PATH, index=False)
    st.info(f"🗒️ Scan logged to: `{LOG_PATH}`")
except Exception as e:
    st.warning(f"⚠️ Could not save log: {e}")
