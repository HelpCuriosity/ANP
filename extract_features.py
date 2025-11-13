# extract_features.py
import pefile
import pandas as pd
from pathlib import Path

def extract_pe_features_from_path(file_path):
    """
    Accepts a path string or Path object to a PE file on disk and returns a DataFrame
    with one row of features compatible with your model.
    """
    file_path = Path(file_path)
    try:
        pe = pefile.PE(str(file_path), fast_load=True)
        pe.parse_data_directories()  # optional, for some detailed fields
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
        raise RuntimeError(f"Failed to extract PE features from {file_path}: {e}")

def extract_pe_features_from_bytes(data_bytes):
    """
    Accepts raw file bytes (useful for web upload) and returns a DataFrame.
    """
    try:
        pe = pefile.PE(data=data_bytes, fast_load=True)
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
        raise RuntimeError(f"Failed to extract PE features from bytes: {e}")
